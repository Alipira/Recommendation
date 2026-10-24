"""Top-K candidates from the graph vectors (dot product over the whole catalogue)."""

from __future__ import annotations

import torch


@torch.no_grad()
def topk_items(
    user_vecs: torch.Tensor,
    item_vecs: torch.Tensor,
    user_rows: torch.Tensor | list[int],
    k: int,
    chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Score `user_rows` against every item and keep the best k.
    Works in chunks so the full user x item matrix never sits in memory.
    Returns (scores, item_indices), both (len(user_rows), k), on CPU.
    """
    rows = torch.as_tensor(user_rows, dtype=torch.long, device=user_vecs.device)
    k = min(k, item_vecs.shape[0])
    all_s, all_i = [], []
    for start in range(0, len(rows), chunk):
        scores = user_vecs[rows[start:start + chunk]] @ item_vecs.T
        s, i = torch.topk(scores, k, dim=1)
        all_s.append(s.cpu())
        all_i.append(i.cpu())
    if not all_s:
        return torch.empty(0, k), torch.empty(0, k, dtype=torch.long)
    return torch.cat(all_s), torch.cat(all_i)
