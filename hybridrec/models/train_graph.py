"""Training loop for LightGCN (BPR loss, early stopping on a validation metric)."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch

from hybridrec.models.lightgcn import GraphConfig, LightGCN


def sample_negatives(
    users: np.ndarray, n_items: int, positive_codes: np.ndarray, rng: np.random.Generator, tries: int = 5
) -> np.ndarray:
    """
    Uniform random items, re-drawn when the user already bought them.
    positive_codes: sorted array of user * n_items + item for all training pairs.
    """
    neg = rng.integers(0, n_items, size=len(users))
    for _ in range(tries):
        codes = users * n_items + neg
        pos = np.searchsorted(positive_codes, codes)
        pos = np.minimum(pos, len(positive_codes) - 1)
        clash = positive_codes[pos] == codes
        if not clash.any():
            break
        neg[clash] = rng.integers(0, n_items, size=int(clash.sum()))
    return neg


def train_graph(
    model: LightGCN,
    train_users: np.ndarray,
    train_items: np.ndarray,
    evaluate: Callable[[LightGCN], float],
    cfg: GraphConfig,
    device: str = "cpu",
    log=print,
) -> dict[str, list[float]]:
    """
    train_users / train_items: index arrays of training pairs (one row per purchase).
    evaluate: returns a validation score where higher is better (e.g. recall@20).
    The best weights (by that score) are loaded back into `model` at the end.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    model.to(device)

    pairs = np.unique(np.stack([train_users, train_items], axis=1), axis=0)
    codes = np.sort(pairs[:, 0] * model.n_items + pairs[:, 1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: dict[str, list[float]] = {"loss": [], "bpr": [], "valid": []}
    best_score, best_state, bad = -np.inf, None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rng.permutation(len(pairs))
        sum_loss = sum_bpr = 0.0
        n_batches = 0
        for start in range(0, len(order), cfg.batch_size):
            batch = pairs[order[start:start + cfg.batch_size]]
            neg = sample_negatives(batch[:, 0], model.n_items, codes, rng)
            u = torch.as_tensor(batch[:, 0], device=device)
            p = torch.as_tensor(batch[:, 1], device=device)
            n = torch.as_tensor(neg, device=device)
            loss, bpr = model.bpr_loss(u, p, n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_loss += loss.item()
            sum_bpr += bpr.item()
            n_batches += 1
        history["loss"].append(sum_loss / n_batches)
        history["bpr"].append(sum_bpr / n_batches)

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            score = evaluate(model)
            history["valid"].append(score)
            log(f"graph epoch {epoch}/{cfg.epochs}  loss={history['loss'][-1]:.4f}  valid={score:.4f}")
            if score > best_score:
                best_score, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    log(f"early stop at epoch {epoch} (best valid={best_score:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
