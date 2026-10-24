"""
Ranking metrics over the full catalogue.

All scores are computed against every item, never a sample of negatives:
sampled metrics can put models in the wrong order (Krichene & Rendle, 2020).

Results are also split into:
  repeat   held-out items the user already bought in training
  explore  held-out items that are new to the user
A model can look strong just by repeating past orders, so both parts are
shown (Li et al., "A Next Basket Recommendation Reality Check").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _dcg(hits: np.ndarray) -> float:
    return float((hits / np.log2(np.arange(2, len(hits) + 2))).sum())


def rank_metrics(
    ranked: Mapping[str, Sequence[str]],
    targets: Mapping[str, set[str]],
    k: int,
    history: Mapping[str, set[str]] | None = None,
) -> dict[str, float]:
    """
    ranked:  user -> items in ranked order (at least k long ideally)
    targets: user -> items in the held-out basket
    history: user -> items bought in training (enables repeat/explore split)

    Users in `targets` but missing from `ranked` count as zero (no silent drop).
    """
    recall, ndcg, hit = [], [], []
    rec_rep, rec_exp = [], []
    for user, truth in targets.items():
        if not truth:
            continue
        top = list(ranked.get(user, []))[:k]
        hits = np.array([it in truth for it in top], dtype=float)
        n_hit = hits.sum()
        recall.append(n_hit / min(len(truth), k))
        ideal = _dcg(np.ones(min(len(truth), k)))
        ndcg.append(_dcg(hits) / ideal if ideal > 0 else 0.0)
        hit.append(float(n_hit > 0))

        if history is not None:
            seen = history.get(user, set())
            rep, exp = truth & seen, truth - seen
            top_set = set(top)
            if rep:
                rec_rep.append(len(top_set & rep) / min(len(rep), k))
            if exp:
                rec_exp.append(len(top_set & exp) / min(len(exp), k))

    out = {
        f"recall@{k}": float(np.mean(recall)) if recall else 0.0,
        f"ndcg@{k}": float(np.mean(ndcg)) if ndcg else 0.0,
        f"hit@{k}": float(np.mean(hit)) if hit else 0.0,
        "n_users": float(len(recall)),
    }
    if history is not None:
        out[f"recall_repeat@{k}"] = float(np.mean(rec_rep)) if rec_rep else 0.0
        out[f"recall_explore@{k}"] = float(np.mean(rec_exp)) if rec_exp else 0.0
    return out


def paired_bootstrap(
    per_user_a: np.ndarray, per_user_b: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[str, float]:
    """
    Is B better than A? Resample users with replacement and count how often
    B's mean beats A's. Returns the mean gain and a 95% interval.
    """
    a, b = np.asarray(per_user_a, float), np.asarray(per_user_b, float)
    if a.shape != b.shape:
        raise ValueError("both arrays must hold the same users in the same order")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    gains = (b[idx] - a[idx]).mean(axis=1)
    return {
        "gain": float((b - a).mean()),
        "ci_low": float(np.percentile(gains, 2.5)),
        "ci_high": float(np.percentile(gains, 97.5)),
        "p_b_not_better": float((gains <= 0).mean()),
    }
