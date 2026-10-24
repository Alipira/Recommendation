"""
Simple baselines. Every new model must beat these to be worth its cost.

popular   the k most bought items overall, same list for everyone
repeat    the user's own most bought items, topped up with popular ones
"""

from __future__ import annotations

import pandas as pd


def popular_ranking(train: pd.DataFrame, users: list[str], k: int) -> dict[str, list[str]]:
    top = train["item_id"].value_counts().index[:k].tolist()
    return {u: top for u in users}


def repeat_ranking(train: pd.DataFrame, users: list[str], k: int) -> dict[str, list[str]]:
    popular = train["item_id"].value_counts().index[: k * 2].tolist()
    counts = (
        train.groupby(["user_id", "item_id"]).agg(n=("basket_id", "size"), last=("ts", "max"))
        .reset_index()
        .sort_values(["user_id", "n", "last"], ascending=[True, False, False])
    )
    own = counts.groupby("user_id")["item_id"].apply(list).to_dict()
    out = {}
    for u in users:
        lst = own.get(u, [])[:k]
        seen = set(lst)
        lst += [it for it in popular if it not in seen][: k - len(lst)]
        out[u] = lst
    return out
