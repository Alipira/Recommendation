"""
Instacart Market Basket Analysis adapter.

Download the dataset (Kaggle: "instacart-market-basket-analysis") and point
`root` at the folder that holds:

    orders.csv
    order_products__prior.csv
    order_products__train.csv
    products.csv
    aisles.csv
    departments.csv

Instacart has no dates, only the order number and the days since the previous
order. We rebuild a time axis per user by adding up those gaps, so the first
order of each user is day 0.

The "test" orders in orders.csv have no products, so they are dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hybridrec.data.schema import Dataset, normalise_interactions


def load_instacart(
    root: str | Path,
    max_users: int | None = None,
    seed: int = 0,
) -> Dataset:
    """
    Load Instacart into the data contract.

    max_users: keep a random sample of users (handy for quick runs; the full
    set has ~206k users and ~33M rows).
    """
    root = Path(root)
    orders = pd.read_csv(
        root / "orders.csv",
        usecols=["order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"],
    )
    orders = orders[orders["eval_set"] != "test"]

    if max_users is not None:
        users = orders["user_id"].unique()
        rng = np.random.default_rng(seed)
        keep = rng.choice(users, size=min(max_users, len(users)), replace=False)
        orders = orders[orders["user_id"].isin(keep)]

    orders = orders.sort_values(["user_id", "order_number"])
    gaps = orders["days_since_prior_order"].fillna(0.0)
    orders["ts"] = gaps.groupby(orders["user_id"]).cumsum()
    # Same-day orders get a tiny offset so their order is kept.
    orders["ts"] += orders["order_number"] * 1e-3

    lines = pd.concat(
        [
            pd.read_csv(root / f"order_products__{part}.csv", usecols=["order_id", "product_id"])
            for part in ("prior", "train")
        ],
        ignore_index=True,
    )
    lines = lines.merge(orders[["order_id", "user_id", "ts"]], on="order_id", how="inner")

    interactions = normalise_interactions(
        pd.DataFrame(
            {
                "user_id": lines["user_id"],
                "item_id": lines["product_id"],
                "basket_id": lines["order_id"],
                "ts": lines["ts"],
                "qty": 1.0,
            }
        )
    )

    products = pd.read_csv(root / "products.csv")
    aisles = pd.read_csv(root / "aisles.csv")
    departments = pd.read_csv(root / "departments.csv")
    items = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")
    items = pd.DataFrame(
        {
            "item_id": items["product_id"].astype(str),
            "title": items["product_name"].astype(str),
            "aisle": items["aisle"].astype(str),
            "department": items["department"].astype(str),
        }
    )

    return Dataset(
        interactions=interactions,
        items=items,
        users=None,  # built from history later, see features.history_user_features
        item_text_cols=["aisle", "department"],
        name="instacart",
    ).validate()
