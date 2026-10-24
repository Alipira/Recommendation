"""
LLM reranker: Qwen + LoRA, fed with graph vectors as soft tokens.

What is trained:
  - two small MLPs (projectors) that map graph vectors to the LLM's hidden size
  - LoRA adapters on the attention layers
The base LLM stays frozen.

Loss: the held-out basket can hold several of the candidates, so the loss
pushes up the total probability of all bought candidates:
    loss = -log( sum over bought candidates of p(letter) )
Samples where no candidate was bought are skipped (nothing to learn), and
the share of such samples is reported as "coverage" (the retrieval ceiling).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from hybridrec.llm.prompt import ITEM_TOKEN, LETTERS, USER_TOKEN


@dataclass
class RerankerConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    projector_hidden: int = 512
    dtype: str = "bfloat16"         # "float32" on CPU
    gradient_checkpointing: bool = False


class Projector(nn.Module):
    """Graph vector -> LLM embedding (MLP as in LLaRA)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Sample:
    prompt: str
    user_row: int                 # row in the user vector table
    item_rows: list[int]          # row in the item vector table, one per candidate, in prompt order
    positives: list[int] = field(default_factory=list)  # candidate positions that were bought
    candidate_ids: list[str] = field(default_factory=list)  # item ids, same order as item_rows


class GraphLLMReranker(nn.Module):
    def __init__(self, llm: nn.Module, tokenizer, graph_dim: int, cfg: RerankerConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        added = tokenizer.add_special_tokens({"additional_special_tokens": [USER_TOKEN, ITEM_TOKEN]})
        if added:
            llm.resize_token_embeddings(len(tokenizer))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        self.user_tok = tokenizer.convert_tokens_to_ids(USER_TOKEN)
        self.item_tok = tokenizer.convert_tokens_to_ids(ITEM_TOKEN)
        self.letter_ids = self._letter_token_ids(tokenizer)

        from peft import LoraConfig, get_peft_model

        if cfg.gradient_checkpointing:
            llm.gradient_checkpointing_enable()
            llm.enable_input_require_grads()
        lora = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets, task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(llm, lora)

        d_llm = self.llm.get_input_embeddings().embedding_dim
        self.user_proj = Projector(graph_dim, cfg.projector_hidden, d_llm)
        self.item_proj = Projector(graph_dim, cfg.projector_hidden, d_llm)

    # ── setup helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _letter_token_ids(tokenizer) -> list[int]:
        """Token id for each answer letter. Each letter must be one token."""
        ids = []
        for letter in LETTERS:
            for text in (" " + letter, letter):
                enc = tokenizer.encode(text, add_special_tokens=False)
                if len(enc) == 1:
                    ids.append(enc[0])
                    break
            else:
                raise ValueError(f"letter {letter!r} is not a single token for this tokenizer")
        if len(set(ids)) != len(ids):
            raise ValueError("answer letters map to repeated token ids")
        return ids

    @classmethod
    def from_pretrained(cls, graph_dim: int, cfg: RerankerConfig, device: str = "cuda") -> "GraphLLMReranker":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        llm = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=getattr(torch, cfg.dtype))
        return cls(llm, tok, graph_dim, cfg).to(device)

    # ── forward ─────────────────────────────────────────────────────────────
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _embed(self, samples: list[Sample], user_vecs: torch.Tensor, item_vecs: torch.Tensor):
        device = user_vecs.device
        enc = self.tokenizer([s.prompt for s in samples], return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        emb_layer = self.llm.get_input_embeddings()
        embeds = emb_layer(ids)
        dtype = embeds.dtype

        # Swap special-token embeddings for projected graph vectors.
        u_pos = (ids == self.user_tok).nonzero(as_tuple=False)
        i_pos = (ids == self.item_tok).nonzero(as_tuple=False)
        u_rows = torch.tensor([s.user_row for s in samples], device=device)
        i_rows = torch.tensor([r for s in samples for r in s.item_rows], device=device)
        if len(u_pos) != len(samples) or len(i_pos) != len(i_rows):
            raise ValueError("prompt special tokens do not match the given user/item rows")

        embeds = embeds.clone()
        embeds[u_pos[:, 0], u_pos[:, 1]] = self.user_proj(user_vecs[u_rows]).to(dtype)
        embeds[i_pos[:, 0], i_pos[:, 1]] = self.item_proj(item_vecs[i_rows]).to(dtype)
        return embeds, attn

    def candidate_logits(
        self, samples: list[Sample], user_vecs: torch.Tensor, item_vecs: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns (batch, max_candidates) logits; slots past a sample's
        candidate count are -inf.
        """
        embeds, attn = self._embed(samples, user_vecs, item_vecs)
        out = self.llm(inputs_embeds=embeds, attention_mask=attn)
        last = attn.sum(dim=1) - 1                                     # right padding
        final = out.logits[torch.arange(len(samples), device=attn.device), last]   # (B, vocab)
        n_max = max(len(s.item_rows) for s in samples)
        letters = torch.tensor(self.letter_ids[:n_max], device=final.device)
        logits = final[:, letters].float()
        counts = torch.tensor([len(s.item_rows) for s in samples], device=final.device)
        mask = torch.arange(n_max, device=final.device)[None, :] >= counts[:, None]
        return logits.masked_fill(mask, float("-inf"))

    @staticmethod
    def multi_positive_loss(logits: torch.Tensor, samples: list[Sample]) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        pos = torch.zeros_like(logp, dtype=torch.bool)
        for b, s in enumerate(samples):
            pos[b, s.positives] = True
        return -torch.logsumexp(logp.masked_fill(~pos, float("-inf")), dim=1).mean()

    # ── save / load ─────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(path / "lora")
        torch.save({"user_proj": self.user_proj.state_dict(), "item_proj": self.item_proj.state_dict()},
                   path / "projectors.pt")
        (path / "reranker_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    def load_trained(self, path: str | Path) -> None:
        """Load projectors and LoRA weights saved by save()."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        path = Path(path)
        proj = torch.load(path / "projectors.pt", map_location="cpu")
        self.user_proj.load_state_dict(proj["user_proj"])
        self.item_proj.load_state_dict(proj["item_proj"])
        set_peft_model_state_dict(self.llm, load_file(str(path / "lora" / "adapter_model.safetensors")))
