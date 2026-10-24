"""
Customer features for the VAE.

When the data source has no customer table (public datasets), we describe
each customer by their own order history. Only the training split is used,
so nothing from validation or test leaks in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def history_user_features(
    train: pd.DataFrame,
    items: pd.DataFrame,
    category_col: str | None = None,
    top_categories: int = 20,
) -> pd.DataFrame:
    """
    One row per user with:
      n_baskets, mean_basket_size, mean_gap_days, repeat_ratio, n_distinct_items,
      and the share of purchases in each of the `top_categories` largest
      categories (if category_col is given).
    Counts are log1p-scaled so a few heavy buyers do not dominate.
    """
    baskets = train.groupby(["user_id", "basket_id"]).agg(ts=("ts", "first"), size=("item_id", "size"))
    baskets = baskets.reset_index().sort_values(["user_id", "ts"])
    baskets["gap"] = baskets.groupby("user_id")["ts"].diff()

    per_user = baskets.groupby("user_id").agg(
        n_baskets=("basket_id", "size"),
        mean_basket_size=("size", "mean"),
        mean_gap_days=("gap", "mean"),
    )
    pairs = train.groupby(["user_id", "item_id"]).size()
    per_user["n_distinct_items"] = pairs.groupby("user_id").size()
    # Share of purchases that were of an item bought before (by this user).
    per_user["repeat_ratio"] = ((pairs - 1).groupby("user_id").sum() / pairs.groupby("user_id").sum())

    per_user["mean_gap_days"] = per_user["mean_gap_days"].fillna(per_user["mean_gap_days"].median())
    for col in ("n_baskets", "mean_basket_size", "n_distinct_items", "mean_gap_days"):
        per_user[col] = np.log1p(per_user[col].fillna(0.0))

    if category_col is not None:
        cat = train[["user_id", "item_id"]].merge(
            items[["item_id", category_col]], on="item_id", how="left"
        )
        top = cat[category_col].value_counts().index[:top_categories]
        cat = cat[cat[category_col].isin(top)]
        shares = pd.crosstab(cat["user_id"], cat[category_col], normalize="index")
        shares.columns = [f"share_{c}" for c in shares.columns]
        per_user = per_user.join(shares, how="left")

    per_user = per_user.fillna(0.0).astype(np.float32)
    return per_user.reset_index()


def standardise(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Z-score every numeric column. Returns the table and the (mean, std) used."""
    out = features.copy()
    stats: dict[str, tuple[float, float]] = {}
    for col in out.columns:
        if col == "user_id":
            continue
        mu, sd = float(out[col].mean()), float(out[col].std(ddof=0))
        sd = sd if sd > 1e-8 else 1.0
        out[col] = ((out[col] - mu) / sd).astype(np.float32)
        stats[col] = (mu, sd)
    return out, stats
