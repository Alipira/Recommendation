"""
Prompt layout for the reranker.

Example (titles shortened):

    Customer: <|user_vec|>
    Recent purchases, newest first: Fresh Milk 1L; White Bread; Tea 100g
    Candidates:
    A. Fresh Milk 1L <|item_vec|>
    B. Cheddar Cheese 200g <|item_vec|>
    ...
    Which candidate is the customer most likely to buy next?
    Answer:

<|user_vec|> and <|item_vec|> are special tokens. Before the model runs,
their embeddings are replaced with the projected graph vectors of the
customer and of each candidate. So the LLM sees both the words and the
purchase-graph position of every product.

The answer is read from the next-token logits over the letters A, B, C, ...
One forward pass ranks all candidates; nothing is generated.
"""

from __future__ import annotations

from dataclasses import dataclass

USER_TOKEN = "<|user_vec|>"
ITEM_TOKEN = "<|item_vec|>"
LETTERS = [chr(ord("A") + i) for i in range(26)]


@dataclass
class PromptConfig:
    max_history: int = 10      # recent distinct items shown as text
    max_title_chars: int = 60  # long titles are cut to keep prompts short


def _short(title: str, n: int) -> str:
    title = " ".join(str(title).split())
    return title if len(title) <= n else title[: n - 1].rstrip() + "…"


def build_prompt(
    history_titles: list[str],
    candidate_titles: list[str],
    cfg: PromptConfig = PromptConfig(),
) -> str:
    if len(candidate_titles) > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} candidates are supported")
    hist = "; ".join(_short(t, cfg.max_title_chars) for t in history_titles[: cfg.max_history]) or "none"
    lines = [
        f"Customer: {USER_TOKEN}",
        f"Recent purchases, newest first: {hist}",
        "Candidates:",
    ]
    for letter, title in zip(LETTERS, candidate_titles):
        lines.append(f"{letter}. {_short(title, cfg.max_title_chars)} {ITEM_TOKEN}")
    lines.append("Which candidate is the customer most likely to buy next?")
    lines.append("Answer:")
    return "\n".join(lines)
