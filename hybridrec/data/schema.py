"""
The data contract every adapter must return.

Three tables, all pandas DataFrames:

interactions   one row per (user, item) in a basket
    user_id    str
    item_id    str
    basket_id  str     orders / sessions; all rows of one basket share it
    ts         float   time of the basket, in days (any origin)
    qty        float   quantity, 1.0 if unknown

items          one row per product in the catalogue
    item_id    str
    title      str     short product name, used in the LLM prompt
    + any text attribute columns listed in Dataset.item_text_cols

users          optional, one row per user
    user_id    str
    + numeric feature columns, used as VAE input

Adapters (Instacart, ERP, ...) only need to produce these tables. Everything
after that (splits, graph, models, evaluation) works the same for all sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INTERACTION_COLS = ["user_id", "item_id", "basket_id", "ts", "qty"]


class SchemaError(ValueError):
    """Raised when a table does not follow the data contract."""


@dataclass
class Dataset:
    interactions: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame | None = None
    item_text_cols: list[str] = field(default_factory=list)
    name: str = "dataset"

    def validate(self) -> "Dataset":
        """Check columns, types and links between tables. Returns self."""
        inter, items = self.interactions, self.items

        missing = [c for c in INTERACTION_COLS if c not in inter.columns]
        if missing:
            raise SchemaError(f"interactions is missing columns: {missing}")
        for col in ("item_id", "title"):
            if col not in items.columns:
                raise SchemaError(f"items is missing column '{col}'")
        for col in self.item_text_cols:
            if col not in items.columns:
                raise SchemaError(f"item_text_cols names unknown column '{col}'")

        if items["item_id"].duplicated().any():
            raise SchemaError("items has duplicate item_id values")
        if inter[["user_id", "item_id", "basket_id"]].isna().any().any():
            raise SchemaError("interactions has empty ids")
        if not np.isfinite(inter["ts"].to_numpy(dtype=float)).all():
            raise SchemaError("interactions.ts must be finite numbers")

        unknown = ~inter["item_id"].isin(items["item_id"])
        if unknown.any():
            examples = inter.loc[unknown, "item_id"].unique()[:5].tolist()
            raise SchemaError(f"interactions has items not in the catalogue: {examples}")

        # A basket belongs to one user at one time.
        per_basket = inter.groupby("basket_id").agg(
            n_users=("user_id", "nunique"), n_ts=("ts", "nunique")
        )
        if (per_basket["n_users"] > 1).any() or (per_basket["n_ts"] > 1).any():
            raise SchemaError("each basket_id must have exactly one user and one ts")

        if self.users is not None:
            if "user_id" not in self.users.columns:
                raise SchemaError("users is missing column 'user_id'")
            if self.users["user_id"].duplicated().any():
                raise SchemaError("users has duplicate user_id values")
            feats = self.users.drop(columns="user_id")
            non_numeric = [c for c in feats.columns if not pd.api.types.is_numeric_dtype(feats[c])]
            if non_numeric:
                raise SchemaError(f"users feature columns must be numeric: {non_numeric}")
        return self

    def item_text(self) -> dict[str, str]:
        """item_id -> 'title (attr1, attr2)' used in prompts."""
        items = self.items
        if not self.item_text_cols:
            return dict(zip(items["item_id"], items["title"].astype(str)))
        extra = items[self.item_text_cols].astype(str).agg(", ".join, axis=1)
        text = items["title"].astype(str) + " (" + extra + ")"
        return dict(zip(items["item_id"], text))


def normalise_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cast to the contract types and drop exact duplicate rows (summing qty)."""
    out = df[INTERACTION_COLS].copy()
    for col in ("user_id", "item_id", "basket_id"):
        out[col] = out[col].astype(str)
    out["ts"] = out["ts"].astype(float)
    out["qty"] = out["qty"].fillna(1.0).astype(float)
    out = out.groupby(["user_id", "item_id", "basket_id", "ts"], as_index=False)["qty"].sum()
    return out[INTERACTION_COLS]
