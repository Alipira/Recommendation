import copy

import pytest
import torch

from hybridrec.llm.prompt import ITEM_TOKEN, USER_TOKEN, build_prompt
from hybridrec.llm.reranker import GraphLLMReranker, RerankerConfig, Sample
from hybridrec.llm.tiny import make_tiny_llm
from hybridrec.llm.train_reranker import RerankTrainConfig, make_samples, rerank, train_reranker
from hybridrec.data.indexer import Indexer
import pandas as pd

TITLES = ["Fresh Milk 1", "Cheese 2", "Tea 3", "Bread 4"]
CFG = RerankerConfig(dtype="float32", lora_r=4, lora_alpha=8, projector_hidden=16)


@pytest.fixture()
def setup():
    torch.manual_seed(0)
    p4 = build_prompt(["Fresh Milk 1"], TITLES)
    p2 = build_prompt([], TITLES[:2])
    llm, tok = make_tiny_llm([p4, p2])
    base = copy.deepcopy(llm)
    model = GraphLLMReranker(llm, tok, graph_dim=8, cfg=CFG)
    samples = [
        Sample(p4, 0, [0, 1, 2, 3], [1], ["a", "b", "c", "d"]),
        Sample(p2, 2, [4, 0], [0], ["e", "a"]),
    ]
    return model, base, tok, samples, torch.randn(3, 8), torch.randn(5, 8)


def test_prompt_has_one_token_per_candidate():
    p = build_prompt(["x"], TITLES)
    assert p.count(USER_TOKEN) == 1 and p.count(ITEM_TOKEN) == 4
    assert p.rstrip().endswith("Answer:")
    with pytest.raises(ValueError):
        build_prompt([], ["t"] * 27)


def test_logits_mask_unused_letters(setup):
    model, _, _, samples, uv, iv = setup
    logits = model.candidate_logits(samples, uv, iv)
    assert logits.shape == (2, 4)
    assert torch.isinf(logits[1, 2:]).all() and torch.isfinite(logits[1, :2]).all()


def test_graph_vectors_change_the_output(setup):
    model, _, _, samples, uv, iv = setup
    model.eval()
    a = model.candidate_logits(samples, uv, iv)
    b = model.candidate_logits(samples, uv, iv + 1.0)
    assert not torch.allclose(a, b)


def test_mismatched_rows_raise(setup):
    model, _, _, samples, uv, iv = setup
    bad = [Sample(samples[0].prompt, 0, [0, 1], [0], ["a", "b"])]  # prompt has 4 item tokens
    with pytest.raises(ValueError, match="special tokens"):
        model.candidate_logits(bad, uv, iv)


def test_multi_positive_loss():
    logits = torch.tensor([[2.0, 0.0, 0.0]])
    s = [Sample("", 0, [0, 1, 2], [0, 1])]
    p = torch.softmax(logits, 1)[0]
    expected = -torch.log(p[0] + p[1])
    assert GraphLLMReranker.multi_positive_loss(logits, s).item() == pytest.approx(expected.item(), rel=1e-5)


def test_only_lora_and_projectors_train(setup):
    model = setup[0]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n or "_proj.net" in n for n in names)


def test_training_lowers_loss(setup):
    model, _, _, samples, uv, iv = setup
    before = model.multi_positive_loss(model.candidate_logits(samples, uv, iv), samples).item()
    cfg = RerankTrainConfig(epochs=40, batch_size=2, grad_accum=1, lr=1e-2, eval_every_steps=1000)
    train_reranker(model, samples, uv, iv, cfg, log=lambda *_: None)
    model.eval()
    after = model.multi_positive_loss(model.candidate_logits(samples, uv, iv), samples).item()
    assert after < before


def test_save_and_load_give_same_scores(setup, tmp_path):
    model, base, tok, samples, uv, iv = setup
    train_reranker(model, samples, uv, iv, RerankTrainConfig(epochs=5, batch_size=2, grad_accum=1, lr=1e-2,
                                                             eval_every_steps=1000), log=lambda *_: None)
    model.eval()
    expected = model.candidate_logits(samples, uv, iv)
    model.save(tmp_path / "rr")

    fresh = GraphLLMReranker(base, tok, graph_dim=8, cfg=CFG)
    fresh.load_trained(tmp_path / "rr")
    fresh.eval()
    assert torch.allclose(fresh.candidate_logits(samples, uv, iv), expected, atol=1e-5)


def test_make_samples_and_rerank(setup):
    model, _, _, _, uv, iv = setup
    ix = Indexer(["u0", "u1", "u2"], ["a", "b", "c", "d", "e"])
    text = {k: t for k, t in zip("abcd", TITLES)}
    cands = {"u0": ["a", "b", "zzz"], "u1": ["c", "d"], "unknown": ["a"]}
    samples, kept, stats = make_samples(["u0", "u1", "unknown"], cands, {}, text, ix, {"u0": {"b"}, "u1": {"x"}})
    assert kept == ["u0"] and samples[0].positives == [1] and samples[0].candidate_ids == ["a", "b"]
    assert stats["coverage"] == 0.5
    out = rerank(model, samples, kept, uv, iv)
    assert sorted(out["u0"]) == ["a", "b"]
