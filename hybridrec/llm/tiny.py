"""
A tiny random Qwen3 model with a word-level tokenizer, built offline.

Only for tests and smoke runs: it checks that the whole pipeline runs and
that the loss goes down. Its rankings mean nothing.
"""

from __future__ import annotations

from collections.abc import Iterable


def make_tiny_llm(texts: Iterable[str], hidden: int = 64, layers: int = 2):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()]
    )
    trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]"])
    letters = [chr(ord("A") + i) for i in range(26)]
    tok.train_from_iterator(list(texts) + [" ".join(letters)], trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )

    cfg = Qwen3Config(
        vocab_size=len(tokenizer) + 8,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg), tokenizer
