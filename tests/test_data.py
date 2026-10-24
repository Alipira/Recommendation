import pandas as pd
import pytest

from hybridrec.data.features import history_user_features, standardise
from hybridrec.data.indexer import Indexer
from hybridrec.data.schema import Dataset, SchemaError, normalise_interactions
from hybridrec.data.splits import Split, last_basket_split, time_cutoff_split
from hybridrec.data.synthetic import make_synthetic


def _inter(rows):
    return normalise_interactions(pd.DataFrame(rows, columns=["user_id", "item_id", "basket_id", "ts", "qty"]))


ITEMS = pd.DataFrame({"item_id": ["a", "b", "c"], "title": ["A", "B", "C"], "cat": ["x", "y", "x"]})


def test_valid_dataset_passes():
    inter = _inter([("u1", "a", "b1", 0, 1), ("u1", "b", "b1", 0, 2)])
    Dataset(inter, ITEMS, item_text_cols=["cat"]).validate()


def test_unknown_item_is_rejected():
    inter = _inter([("u1", "zzz", "b1", 0, 1)])
    with pytest.raises(SchemaError, match="not in the catalogue"):
        Dataset(inter, ITEMS).validate()


def test_basket_with_two_users_is_rejected():
    inter = _inter([("u1", "a", "b1", 0, 1), ("u2", "b", "b1", 0, 1)])
    with pytest.raises(SchemaError, match="exactly one user"):
        Dataset(inter, ITEMS).validate()


def test_non_numeric_user_features_rejected():
    inter = _inter([("u1", "a", "b1", 0, 1)])
    users = pd.DataFrame({"user_id": ["u1"], "segment": ["gold"]})
    with pytest.raises(SchemaError, match="numeric"):
        Dataset(inter, ITEMS, users=users).validate()


def test_normalise_sums_duplicate_rows():
    inter = _inter([("u1", "a", "b1", 0, 1), ("u1", "a", "b1", 0, 2)])
    assert len(inter) == 1 and inter["qty"].iloc[0] == 3


def test_item_text_includes_attributes():
    text = Dataset(_inter([("u1", "a", "b1", 0, 1)]), ITEMS, item_text_cols=["cat"]).item_text()
    assert text["a"] == "A (x)"


def test_last_basket_split_holds_out_latest_baskets():
    rows = [("u1", "a", f"b{t}", float(t), 1) for t in range(4)] + [("u2", "a", "c0", 0.0, 1)]
    s = last_basket_split(_inter(rows), min_baskets=3)
    assert set(s.test["basket_id"]) == {"b3"}
    assert set(s.valid["basket_id"]) == {"b2"}
    assert set(s.train["basket_id"]) == {"b0", "b1", "c0"}  # u2 has too few baskets: train only


def test_split_has_no_overlap_and_no_future_leak():
    ds = make_synthetic(n_users=50, seed=1)
    s = last_basket_split(ds.interactions)
    b = [set(df["basket_id"]) for df in (s.train, s.valid, s.test)]
    assert not (b[0] & b[1] or b[0] & b[2] or b[1] & b[2])
    assert sum(len(x) for x in (s.train, s.valid, s.test)) == len(ds.interactions)
    last_train = s.train.groupby("user_id")["ts"].max()
    first_test = s.test.groupby("user_id")["ts"].min()
    assert (last_train.reindex(first_test.index) < first_test).all()


def test_time_cutoff_split():
    rows = [("u1", "a", "b0", 1.0, 1), ("u1", "b", "b1", 5.0, 1), ("u1", "c", "b2", 9.0, 1),
            ("u9", "a", "x", 6.0, 1)]  # u9 never appears in train
    s = time_cutoff_split(_inter(rows), valid_start=4, test_start=8)
    assert list(s.valid["user_id"]) == ["u1"] and list(s.test["basket_id"]) == ["b2"]
    with pytest.raises(ValueError):
        time_cutoff_split(_inter(rows), 8, 4)


def test_targets():
    t = Split.targets(_inter([("u1", "a", "b", 0, 1), ("u1", "b", "b", 0, 1)]))
    assert t == {"u1": {"a", "b"}}


def test_indexer_covers_whole_catalogue_and_round_trips(tmp_path):
    train = _inter([("u2", "a", "b1", 0, 1), ("u1", "b", "b2", 0, 1)])
    ix = Indexer.fit(train, ITEMS)
    assert ix.n_items == 3 and ix.user_ids == ["u1", "u2"]
    u, i = ix.encode(pd.concat([train, _inter([("new", "a", "z", 0, 1)])]))
    assert len(u) == 2  # unknown user dropped
    ix.save(tmp_path / "ix.json")
    assert Indexer.load(tmp_path / "ix.json").item_ids == ix.item_ids


def test_history_features_are_finite_and_standardised():
    ds = make_synthetic(n_users=40, seed=2)
    f = history_user_features(ds.interactions, ds.items, "category")
    assert f["user_id"].is_unique and f.drop(columns="user_id").notna().all().all()
    z, stats = standardise(f)
    assert abs(z["n_baskets"].mean()) < 1e-5 and "n_baskets" in stats
