import math

import numpy as np
import pandas as pd
import pytest

from hybridrec.candidates import merge_candidates
from hybridrec.eval.baselines import popular_ranking, repeat_ranking
from hybridrec.eval.metrics import paired_bootstrap, rank_metrics


def test_perfect_ranking_scores_one():
    m = rank_metrics({"u": ["a", "b", "c"]}, {"u": {"a", "b"}}, k=2)
    assert m["recall@2"] == 1.0 and m["ndcg@2"] == pytest.approx(1.0) and m["hit@2"] == 1.0


def test_ndcg_by_hand():
    # hit at rank 2 only, one relevant item: dcg = 1/log2(3), idcg = 1
    m = rank_metrics({"u": ["x", "a", "y"]}, {"u": {"a"}}, k=3)
    assert m["ndcg@3"] == pytest.approx(1 / math.log2(3))


def test_missing_user_counts_as_zero():
    m = rank_metrics({"u": ["a"]}, {"u": {"a"}, "v": {"a"}}, k=1)
    assert m["recall@1"] == 0.5 and m["n_users"] == 2


def test_repeat_explore_split():
    m = rank_metrics({"u": ["old", "x"]}, {"u": {"old", "new"}}, k=2, history={"u": {"old"}})
    assert m["recall_repeat@2"] == 1.0 and m["recall_explore@2"] == 0.0


def test_paired_bootstrap_detects_clear_gain():
    a = np.zeros(200)
    b = np.full(200, 0.1)
    r = paired_bootstrap(a, b, n_boot=200)
    assert r["gain"] == pytest.approx(0.1) and r["ci_low"] > 0 and r["p_b_not_better"] == 0


def test_baselines():
    train = pd.DataFrame({
        "user_id": ["u", "u", "u", "v"], "item_id": ["a", "a", "b", "c"],
        "basket_id": ["1", "2", "2", "3"], "ts": [0, 1, 1, 0], "qty": 1.0,
    })
    assert popular_ranking(train, ["u"], 1) == {"u": ["a"]}
    rep = repeat_ranking(train, ["u", "new"], 3)
    assert rep["u"][:2] == ["a", "b"] and len(rep["u"]) == 3
    assert rep["new"][0] == "a"  # no history: popular items


def test_merge_candidates_order_and_dedup():
    out = merge_candidates(["u"], {"u": ["g1", "r1", "g2", "g3"]}, {"u": ["r1", "r2"]}, n_candidates=4, n_repeat=2)
    assert out["u"] == ["r1", "r2", "g1", "g2"]
    with pytest.raises(ValueError):
        merge_candidates(["u"], {}, {}, n_candidates=2, n_repeat=3)
