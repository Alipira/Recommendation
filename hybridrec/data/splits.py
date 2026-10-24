"""
Train / validation / test splits.

Last-basket split (default): for each user, the last basket is test, the one
before it is validation, and the rest is training. This matches how the
system is used: given everything up to now, predict the next order.

Time-cutoff split: everything before `valid_start` is train, between
`valid_start` and `test_start` is validation, the rest is test. Use it when
you want one global "now" for all users (closer to a real deployment).

Never split at random: a random split leaks the future into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    @staticmethod
    def targets(df: pd.DataFrame) -> dict[str, set[str]]:
        """user_id -> set of item_ids in that user's held-out basket(s)."""
        return {u: set(g) for u, g in df.groupby("user_id")["item_id"]}


def last_basket_split(interactions: pd.DataFrame, min_baskets: int = 3) -> Split:
    """
    Hold out each user's last basket (test) and second-to-last (valid).

    Users with fewer than `min_baskets` baskets stay fully in train, because
    they do not have enough history to both learn from and test on.
    """
    baskets = (
        interactions[["user_id", "basket_id", "ts"]]
        .drop_duplicates("basket_id")
        .sort_values(["user_id", "ts", "basket_id"])
    )
    baskets["rank_from_end"] = baskets.groupby("user_id").cumcount(ascending=False)
    baskets["n_baskets"] = baskets.groupby("user_id")["basket_id"].transform("size")

    eligible = baskets["n_baskets"] >= min_baskets
    test_ids = set(baskets.loc[eligible & (baskets["rank_from_end"] == 0), "basket_id"])
    valid_ids = set(baskets.loc[eligible & (baskets["rank_from_end"] == 1), "basket_id"])

    in_test = interactions["basket_id"].isin(test_ids)
    in_valid = interactions["basket_id"].isin(valid_ids)
    return Split(
        train=interactions[~in_test & ~in_valid].reset_index(drop=True),
        valid=interactions[in_valid].reset_index(drop=True),
        test=interactions[in_test].reset_index(drop=True),
    )


def time_cutoff_split(interactions: pd.DataFrame, valid_start: float, test_start: float) -> Split:
    """Global time cutoffs (same units as `ts`). Only users seen in train are evaluated."""
    if not valid_start < test_start:
        raise ValueError("valid_start must be before test_start")
    ts = interactions["ts"]
    train = interactions[ts < valid_start]
    known = set(train["user_id"])
    valid = interactions[(ts >= valid_start) & (ts < test_start) & interactions["user_id"].isin(known)]
    test = interactions[(ts >= test_start) & interactions["user_id"].isin(known)]
    return Split(
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
