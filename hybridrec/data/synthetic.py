"""
Small fake dataset with real structure, for tests and quick smoke runs.

Each user likes a few categories and has a few "staple" items they reorder
often. Baskets mix staples with new items from the liked categories, so both
repeat and explore behaviour exist and a good model can beat popularity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hybridrec.data.schema import Dataset, normalise_interactions

_WORDS = ["Organic", "Fresh", "Classic", "Light", "Spicy", "Sweet", "Crunchy", "Family"]
_NOUNS = ["Milk", "Bread", "Cheese", "Juice", "Chips", "Tea", "Coffee", "Yogurt", "Biscuits", "Rice"]


def make_synthetic(
    n_users: int = 300,
    n_items: int = 200,
    n_categories: int = 8,
    baskets_per_user: tuple[int, int] = (4, 10),
    basket_size: tuple[int, int] = (3, 8),
    seed: int = 0,
) -> Dataset:
    rng = np.random.default_rng(seed)
    item_cat = rng.integers(0, n_categories, size=n_items)
    popularity = rng.zipf(1.6, size=n_items).astype(float)
    popularity = np.minimum(popularity, 50.0)

    items = pd.DataFrame(
        {
            "item_id": [f"i{j}" for j in range(n_items)],
            "title": [
                f"{_WORDS[rng.integers(len(_WORDS))]} {_NOUNS[j % len(_NOUNS)]} {j}" for j in range(n_items)
            ],
            "category": [f"cat{c}" for c in item_cat],
        }
    )

    rows = []
    basket_no = 0
    for u in range(n_users):
        liked = rng.choice(n_categories, size=2, replace=False)
        pool = np.flatnonzero(np.isin(item_cat, liked))
        weights = popularity[pool] / popularity[pool].sum()
        staples = rng.choice(pool, size=min(3, len(pool)), replace=False, p=weights)
        t = 0.0
        for _ in range(rng.integers(*baskets_per_user, endpoint=True)):
            t += float(rng.integers(3, 15))
            size = int(rng.integers(*basket_size, endpoint=True))
            keep = staples[rng.random(len(staples)) < 0.7]
            n_new = max(size - len(keep), 1)
            new = rng.choice(pool, size=min(n_new, len(pool)), replace=False, p=weights)
            for it in set(keep.tolist()) | set(new.tolist()):
                rows.append((f"u{u}", f"i{it}", f"b{basket_no}", t, 1.0))
            basket_no += 1

    inter = normalise_interactions(pd.DataFrame(rows, columns=["user_id", "item_id", "basket_id", "ts", "qty"]))
    return Dataset(inter, items, users=None, item_text_cols=["category"], name="synthetic").validate()
