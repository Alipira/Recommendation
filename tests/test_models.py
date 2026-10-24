import numpy as np
import pytest
import torch

from hybridrec.data.indexer import Indexer
from hybridrec.data.splits import Split, last_basket_split
from hybridrec.data.synthetic import make_synthetic
from hybridrec.eval.baselines import popular_ranking
from hybridrec.eval.metrics import rank_metrics
from hybridrec.models.lightgcn import GraphConfig, LightGCN, build_norm_adj
from hybridrec.models.train_graph import sample_negatives, train_graph
from hybridrec.models.vae import VAEConfig, train_vae
from hybridrec.retrieval import topk_items


def test_norm_adj_is_symmetric_and_normalised():
    adj = build_norm_adj(np.array([0, 0, 1]), np.array([0, 1, 1]), 2, 2).to_dense()
    assert torch.allclose(adj, adj.T)
    # user0 (deg 2) - item0 (deg 1): 1/sqrt(2*1)
    assert adj[0, 2].item() == pytest.approx(1 / np.sqrt(2))
    assert adj[:2, :2].abs().sum() == 0  # no user-user edges


def test_negative_sampling_avoids_positives():
    rng = np.random.default_rng(0)
    users = np.zeros(1000, dtype=np.int64)
    codes = np.array([0, 1, 2, 3])  # user 0 bought items 0..3 of 5
    neg = sample_negatives(users, 5, codes, rng, tries=100)  # 0.8^100 chance left per row
    assert (neg == 4).all()


def test_vae_shapes_and_loss_falls():
    x = np.random.default_rng(0).normal(size=(200, 6)).astype(np.float32)
    losses = []
    model, z = train_vae(x, VAEConfig(latent_dim=3, epochs=15, batch_size=64), log=losses.append)
    assert z.shape == (200, 3) and np.isfinite(z).all()
    first, last = (float(s.split("loss=")[1].split()[0]) for s in (losses[0], losses[-1]))
    assert last < first


def test_topk_items_chunks_match_full():
    u, i = torch.randn(10, 4), torch.randn(30, 4)
    s1, i1 = topk_items(u, i, list(range(10)), 5, chunk=3)
    s2, i2 = torch.topk(u @ i.T, 5)
    assert torch.equal(i1, i2)


@pytest.mark.parametrize("with_features", [False, True])
def test_lightgcn_beats_popularity(with_features):
    ds = make_synthetic(n_users=200, n_items=120, seed=3)
    s = last_basket_split(ds.interactions)
    ix = Indexer.fit(s.train, ds.items)
    u, i = ix.encode(s.train)
    feats = np.random.default_rng(0).normal(size=(ix.n_users, 4)).astype(np.float32) if with_features else None
    model = LightGCN(ix.n_users, ix.n_items, GraphConfig(dim=16, epochs=25, eval_every=25, batch_size=512),
                     user_features=feats)
    model.set_graph(build_norm_adj(u, i, ix.n_users, ix.n_items))

    targets = Split.targets(s.valid)
    users = list(targets)

    def ranked(m):
        fu, fi = m.final_vectors()
        _, idx = topk_items(fu, fi, [ix.user_index[x] for x in users], 10)
        return {x: [ix.item_ids[j] for j in r] for x, r in zip(users, idx.tolist())}

    train_graph(model, u, i, lambda m: rank_metrics(ranked(m), targets, 10)["recall@10"], model.cfg,
                log=lambda *_: None)
    gcn = rank_metrics(ranked(model), targets, 10)["recall@10"]
    pop = rank_metrics(popular_ranking(s.train, users, 10), targets, 10)["recall@10"]
    assert gcn > pop
    if with_features:
        assert model.vectors_from_features(feats[:2]).shape == (2, 16)


def test_propagate_needs_graph():
    with pytest.raises(RuntimeError):
        LightGCN(2, 2, GraphConfig(dim=4)).propagate()
