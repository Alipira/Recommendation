"""Build reranker samples, train the reranker, and rerank candidate lists."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from hybridrec.data.indexer import Indexer
from hybridrec.llm.prompt import PromptConfig, build_prompt
from hybridrec.llm.reranker import GraphLLMReranker, Sample


@dataclass
class RerankTrainConfig:
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    eval_every_steps: int = 200   # optimizer steps between validation checks
    patience: int = 3
    seed: int = 0


def recent_history(interactions: pd.DataFrame, item_text: Mapping[str, str], n: int) -> dict[str, list[str]]:
    """user -> titles of their last n distinct items, newest first."""
    df = interactions.sort_values("ts", ascending=False).drop_duplicates(["user_id", "item_id"])
    df = df.groupby("user_id").head(n)
    return {u: [item_text.get(i, i) for i in g] for u, g in df.groupby("user_id", sort=False)["item_id"]}


def make_samples(
    users: Sequence[str],
    candidates: Mapping[str, Sequence[str]],
    history: Mapping[str, list[str]],
    item_text: Mapping[str, str],
    indexer: Indexer,
    targets: Mapping[str, set[str]] | None = None,
    prompt_cfg: PromptConfig = PromptConfig(),
) -> tuple[list[Sample], list[str], dict[str, float]]:
    """
    Returns (samples, users_kept, stats).
    With targets: users whose candidates contain none of the bought items are
    dropped (training only); stats["coverage"] is the share that was kept.
    """
    samples, kept = [], []
    n_total = n_no_pos = 0
    for u in users:
        if u not in indexer.user_index:
            continue
        cands = [c for c in candidates.get(u, []) if c in indexer.item_index]
        if not cands:
            continue
        n_total += 1
        positives: list[int] = []
        if targets is not None:
            truth = targets.get(u, set())
            positives = [j for j, c in enumerate(cands) if c in truth]
            if not positives:
                n_no_pos += 1
                continue
        prompt = build_prompt(history.get(u, []), [item_text.get(c, c) for c in cands], prompt_cfg)
        samples.append(
            Sample(
                prompt=prompt,
                user_row=indexer.user_index[u],
                item_rows=[indexer.item_index[c] for c in cands],
                positives=positives,
                candidate_ids=list(cands),
            )
        )
        kept.append(u)
    coverage = 1.0 - n_no_pos / n_total if n_total else 0.0
    return samples, kept, {"n_users": float(n_total), "coverage": coverage}


@torch.no_grad()
def rerank(
    model: GraphLLMReranker,
    samples: list[Sample],
    users: list[str],
    user_vecs: torch.Tensor,
    item_vecs: torch.Tensor,
    batch_size: int = 16,
) -> dict[str, list[str]]:
    """Order each user's candidates by the reranker's probability."""
    was = model.training
    model.eval()
    out: dict[str, list[str]] = {}
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        logits = model.candidate_logits(batch, user_vecs, item_vecs).cpu()
        for b, u in enumerate(users[start:start + batch_size]):
            n = len(batch[b].item_rows)
            order = torch.argsort(logits[b, :n], descending=True).tolist()
            out[u] = [batch[b].candidate_ids[j] for j in order]
    model.train(was)
    return out


def train_reranker(
    model: GraphLLMReranker,
    train_samples: list[Sample],
    user_vecs: torch.Tensor,
    item_vecs: torch.Tensor,
    cfg: RerankTrainConfig,
    evaluate=None,
    log=print,
) -> dict[str, list[float]]:
    """
    evaluate: optional callable(model) -> float, higher is better (e.g. valid NDCG@10).
    The best LoRA + projector weights by that score are kept.
    """
    if not train_samples:
        raise ValueError("no training samples (check candidate coverage)")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: dict[str, list[float]] = {"loss": [], "valid": []}
    best, best_state, bad, step = -np.inf, None, 0, 0
    model.train()
    stop = False
    for epoch in range(cfg.epochs):
        order = list(range(len(train_samples)))
        random.shuffle(order)
        running, n_run = 0.0, 0
        for i, start in enumerate(range(0, len(order), cfg.batch_size)):
            batch = [train_samples[j] for j in order[start:start + cfg.batch_size]]
            logits = model.candidate_logits(batch, user_vecs, item_vecs)
            loss = model.multi_positive_loss(logits, batch) / cfg.grad_accum
            loss.backward()
            running += loss.item() * cfg.grad_accum
            n_run += 1
            if (i + 1) % cfg.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % cfg.eval_every_steps == 0:
                history["loss"].append(running / n_run)
                msg = f"rerank epoch {epoch + 1} step {step}  loss={running / n_run:.4f}"
                running, n_run = 0.0, 0
                if evaluate is not None:
                    score = evaluate(model)
                    history["valid"].append(score)
                    msg += f"  valid={score:.4f}"
                    if score > best:
                        best, bad = score, 0
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                                      if "lora_" in k or "_proj.net" in k}
                    else:
                        bad += 1
                log(msg)
                if evaluate is not None and bad >= cfg.patience:
                    log(f"early stop at step {step} (best valid={best:.4f})")
                    stop = True
                    break
        opt.zero_grad()
        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return history
