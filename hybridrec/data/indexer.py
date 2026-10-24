"""Map string ids to the 0..N-1 indices that embedding tables need."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Indexer:
    user_ids: list[str]
    item_ids: list[str]

    def __post_init__(self) -> None:
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {it: i for i, it in enumerate(self.item_ids)}

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @classmethod
    def fit(cls, train: pd.DataFrame, items: pd.DataFrame) -> "Indexer":
        """
        Users: everyone seen in training.
        Items: the whole catalogue, so new items without orders still get an
        index (and can be scored through their node features).
        """
        users = sorted(train["user_id"].unique().tolist())
        item_ids = sorted(items["item_id"].astype(str).unique().tolist())
        return cls(users, item_ids)

    def encode(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (user_idx, item_idx) arrays, dropping rows with unknown ids."""
        u = df["user_id"].map(self.user_index)
        i = df["item_id"].map(self.item_index)
        ok = u.notna() & i.notna()
        return u[ok].to_numpy(dtype=np.int64), i[ok].to_numpy(dtype=np.int64)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"users": self.user_ids, "items": self.item_ids}))

    @classmethod
    def load(cls, path: str | Path) -> "Indexer":
        d = json.loads(Path(path).read_text())
        return cls(d["users"], d["items"])
