"""
Candidate lists for the reranker.

Each list mixes two sources:
  repeat  the customer's own most bought items (B2B shops reorder a lot)
  graph   LightGCN top items, which also brings in items new to the customer

Repeat items go first (up to `n_repeat`), then graph items fill the rest,
skipping duplicates. The reranker then decides the final order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def merge_candidates(
    users: Sequence[str],
    graph_ranked: Mapping[str, Sequence[str]],
    repeat_ranked: Mapping[str, Sequence[str]],
    n_candidates: int = 20,
    n_repeat: int = 8,
) -> dict[str, list[str]]:
    if n_repeat > n_candidates:
        raise ValueError("n_repeat cannot be larger than n_candidates")
    out: dict[str, list[str]] = {}
    for u in users:
        lst: list[str] = []
        seen: set[str] = set()
        for it in list(repeat_ranked.get(u, []))[:n_repeat]:
            if it not in seen:
                lst.append(it)
                seen.add(it)
        for it in graph_ranked.get(u, []):
            if len(lst) >= n_candidates:
                break
            if it not in seen:
                lst.append(it)
                seen.add(it)
        out[u] = lst
    return out
