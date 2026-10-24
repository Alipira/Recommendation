"""
End-to-end pipeline.

    python -m hybridrec.pipeline --config configs/instacart.yaml
    python -m hybridrec.pipeline --config configs/instacart.yaml --stages graph rerank evaluate

Stages (each saves its output in run_dir, so later stages can be rerun alone):

    prepare   load data, split, build customer features, index ids
    vae       train the customer VAE (or load ready-made vectors)
    graph     train LightGCN with the VAE vectors as customer node features
    rerank    train the Qwen reranker on validation baskets
    evaluate  compare baselines, graph, candidates and reranker on test baskets

Leak rules:
  - the graph and the VAE only see the training split
  - the reranker learns from validation baskets (the graph never saw them)
  - test prompts and repeat lists may use train + validation (that is the past)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hybridrec.candidates import merge_candidates
from hybridrec.config import RunConfig, config_to_dict, load_config
from hybridrec.data.features import history_user_features, standardise
from hybridrec.data.indexer import Indexer
from hybridrec.data.schema import Dataset
from hybridrec.data.splits import Split, last_basket_split
from hybridrec.eval.baselines import popular_ranking, repeat_ranking
from hybridrec.eval.metrics import rank_metrics
from hybridrec.retrieval import topk_items

STAGES = ["prepare", "vae", "graph", "rerank", "evaluate"]


def log(msg: str) -> None:
    print(msg, flush=True)


def _device(name: str) -> str:
    if name.startswith("cuda") and not torch.cuda.is_available():
        log(f"CUDA not available, using CPU instead of {name}")
        return "cpu"
    return name


class Run:
    """Holds the config and reads/writes stage outputs in run_dir."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.dir = Path(cfg.run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=2, default=str))
        self.device = _device(cfg.train_device)

    def path(self, name: str) -> Path:
        return self.dir / name

    # ── loaded lazily from disk ─────────────────────────────────────────────
    def load_split(self) -> tuple[Split, pd.DataFrame, dict]:
        split = Split(*(pd.read_parquet(self.path(f"{p}.parquet")) for p in ("train", "valid", "test")))
        items = pd.read_parquet(self.path("items.parquet"))
        meta = json.loads(self.path("meta.json").read_text())
        return split, items, meta

    def item_text(self, items: pd.DataFrame, meta: dict) -> dict[str, str]:
        return Dataset(pd.DataFrame(), items, item_text_cols=meta["item_text_cols"]).item_text()


# ─────────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(cfg: RunConfig) -> Dataset:
    d = cfg.data
    if d.source == "synthetic":
        from hybridrec.data.synthetic import make_synthetic

        return make_synthetic(n_users=d.max_users or 300, seed=d.seed)
    if d.source == "instacart":
        from hybridrec.data.instacart import load_instacart

        return load_instacart(d.path, max_users=d.max_users, seed=d.seed)
    if d.source == "erp":
        from hybridrec.data.erp import ErpConfig, load_erp

        url = os.environ.get(d.erp_connection_env)
        if not url:
            raise EnvironmentError(f"set {d.erp_connection_env} to the database URL")
        return load_erp(ErpConfig(connection_url=url))
    raise ValueError(f"unknown data source '{d.source}'")


def stage_prepare(run: Run) -> None:
    cfg = run.cfg
    ds = load_dataset(cfg)
    split = last_basket_split(ds.interactions, min_baskets=cfg.data.min_baskets)
    for name in ("train", "valid", "test"):
        getattr(split, name).to_parquet(run.path(f"{name}.parquet"))
    ds.items.to_parquet(run.path("items.parquet"))

    indexer = Indexer.fit(split.train, ds.items)
    indexer.save(run.path("indexer.json"))

    # Customer features: the source's own table if it has one, else built from history.
    if ds.users is not None:
        feats = ds.users
    else:
        cat = cfg.data.category_col or (ds.item_text_cols[-1] if ds.item_text_cols else None)
        feats = history_user_features(split.train, ds.items, category_col=cat)
    feats, _ = standardise(feats)
    feats.to_parquet(run.path("user_features.parquet"))

    meta = {"item_text_cols": ds.item_text_cols, "name": ds.name}
    run.path("meta.json").write_text(json.dumps(meta))
    log(
        f"prepare: {ds.name}  rows={len(ds.interactions):,}  users={indexer.n_users:,}  "
        f"items={indexer.n_items:,}  valid users={split.valid['user_id'].nunique():,}  "
        f"test users={split.test['user_id'].nunique():,}"
    )


def stage_vae(run: Run) -> None:
    cfg = run.cfg
    indexer = Indexer.load(run.path("indexer.json"))
    if cfg.data.user_vectors:
        from hybridrec.data.erp import load_user_vectors

        vecs = load_user_vectors(cfg.data.user_vectors).set_index("user_id")
        z = vecs.reindex(indexer.user_ids).fillna(0.0).to_numpy(np.float32)
        log(f"vae: loaded ready-made vectors {z.shape} from {cfg.data.user_vectors}")
    else:
        from hybridrec.models.vae import train_vae

        feats = pd.read_parquet(run.path("user_features.parquet")).set_index("user_id")
        x = feats.reindex(indexer.user_ids).fillna(0.0).to_numpy(np.float32)
        model, z = train_vae(x, cfg.vae, device=run.device, log=log)
        torch.save(model.state_dict(), run.path("vae.pt"))
    np.save(run.path("user_vae.npy"), z)


def _graph_ranked(user_vecs, item_vecs, indexer: Indexer, users: list[str], k: int) -> dict[str, list[str]]:
    users = [u for u in users if u in indexer.user_index]
    _, idx = topk_items(user_vecs, item_vecs, [indexer.user_index[u] for u in users], k)
    return {u: [indexer.item_ids[j] for j in row] for u, row in zip(users, idx.tolist())}


def stage_graph(run: Run) -> None:
    from hybridrec.models.lightgcn import LightGCN, build_norm_adj
    from hybridrec.models.train_graph import train_graph

    cfg = run.cfg
    split, _, _ = run.load_split()
    indexer = Indexer.load(run.path("indexer.json"))
    user_feats = np.load(run.path("user_vae.npy")) if cfg.use_vae else None

    u, i = indexer.encode(split.train)
    adj = build_norm_adj(u, i, indexer.n_users, indexer.n_items)
    model = LightGCN(indexer.n_users, indexer.n_items, cfg.graph, user_features=user_feats).to(run.device)
    model.set_graph(adj)

    valid_t = Split.targets(split.valid)
    valid_users = list(valid_t)
    k = cfg.graph.eval_k

    def evaluate(m) -> float:
        fu, fi = m.final_vectors()
        ranked = _graph_ranked(fu, fi, indexer, valid_users, k)
        return rank_metrics(ranked, valid_t, k)[f"recall@{k}"]

    train_graph(model, u, i, evaluate, cfg.graph, device=run.device, log=log)
    fu, fi = model.final_vectors()
    torch.save(model.state_dict(), run.path("graph.pt"))
    torch.save({"users": fu.cpu(), "items": fi.cpu()}, run.path("graph_vectors.pt"))
    log(f"graph: best valid recall@{k} = {evaluate(model):.4f}")


def _candidate_lists(run: Run, split: Split, indexer: Indexer, users: list[str], past: pd.DataFrame):
    cfg = run.cfg
    vecs = torch.load(run.path("graph_vectors.pt"))
    graph = _graph_ranked(vecs["users"], vecs["items"], indexer, users, cfg.candidates.n_candidates)
    repeat = repeat_ranking(past, users, cfg.candidates.n_repeat)
    # repeat_ranking tops up with popular items; keep only the user's own items here
    own = past.groupby("user_id")["item_id"].apply(set).to_dict()
    repeat = {u: [it for it in lst if it in own.get(u, set())] for u, lst in repeat.items()}
    cands = merge_candidates(users, graph, repeat, cfg.candidates.n_candidates, cfg.candidates.n_repeat)
    return cands, graph, vecs


def _build_reranker(run: Run, graph_dim: int, prompts: list[str]):
    from hybridrec.llm.reranker import GraphLLMReranker

    cfg = run.cfg
    if cfg.tiny_llm:
        # The random base must be the same in the rerank and evaluate stages,
        # so it is saved once and reloaded after that.
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

        from hybridrec.llm.tiny import make_tiny_llm

        cfg.reranker.dtype = "float32"
        base = run.path("tiny_base")
        if base.exists():
            llm = AutoModelForCausalLM.from_pretrained(base)
            tok = PreTrainedTokenizerFast.from_pretrained(base)
        else:
            llm, tok = make_tiny_llm(prompts)
            llm.save_pretrained(base)
            tok.save_pretrained(base)
        return GraphLLMReranker(llm, tok, graph_dim, cfg.reranker).to(run.device)
    return GraphLLMReranker.from_pretrained(graph_dim, cfg.reranker, device=run.device)


def stage_rerank(run: Run) -> None:
    from hybridrec.llm.train_reranker import make_samples, recent_history, rerank, train_reranker

    cfg = run.cfg
    split, items, meta = run.load_split()
    indexer = Indexer.load(run.path("indexer.json"))
    item_text = run.item_text(items, meta)
    targets = Split.targets(split.valid)

    rng = np.random.default_rng(cfg.data.seed)
    users = [u for u in targets if u in indexer.user_index]
    rng.shuffle(users)
    n_hold = max(1, int(len(users) * cfg.reranker_valid_share))
    hold_users, fit_users = users[:n_hold], users[n_hold:]
    if cfg.rerank_max_train_users:
        fit_users = fit_users[: cfg.rerank_max_train_users]

    cands, _, vecs = _candidate_lists(run, split, indexer, hold_users + fit_users, split.train)
    history = recent_history(split.train, item_text, cfg.prompt.max_history)

    fit, _, fit_stats = make_samples(fit_users, cands, history, item_text, indexer, targets, cfg.prompt)
    hold, hold_kept, _ = make_samples(hold_users, cands, history, item_text, indexer, None, cfg.prompt)
    log(f"rerank: {len(fit):,} training samples, candidate coverage {fit_stats['coverage']:.1%}")

    model = _build_reranker(run, vecs["users"].shape[1], [s.prompt for s in fit + hold])
    uvec, ivec = vecs["users"].to(run.device), vecs["items"].to(run.device)
    hold_t = {u: targets[u] for u in hold_kept}
    k = cfg.eval_k

    def evaluate(m) -> float:
        ranked = rerank(m, hold, hold_kept, uvec, ivec, batch_size=cfg.rerank_train.batch_size)
        return rank_metrics(ranked, hold_t, k)[f"ndcg@{k}"]

    base = rank_metrics({u: cands[u] for u in hold_kept}, hold_t, k)[f"ndcg@{k}"]
    log(f"rerank: hold-out ndcg@{k} before training (candidate order) = {base:.4f}")
    train_reranker(model, fit, uvec, ivec, cfg.rerank_train, evaluate=evaluate, log=log)
    model.save(run.path("reranker"))
    log(f"rerank: hold-out ndcg@{k} after training = {evaluate(model):.4f}")


def stage_evaluate(run: Run) -> dict:
    cfg = run.cfg
    split, items, meta = run.load_split()
    indexer = Indexer.load(run.path("indexer.json"))
    item_text = run.item_text(items, meta)
    k = cfg.eval_k

    past = pd.concat([split.train, split.valid], ignore_index=True)
    targets = Split.targets(split.test)
    users = [u for u in targets if u in indexer.user_index]
    targets = {u: targets[u] for u in users}
    seen = past.groupby("user_id")["item_id"].apply(set).to_dict()

    cands, graph, vecs = _candidate_lists(run, split, indexer, users, past)
    results = {
        "popular": rank_metrics(popular_ranking(past, users, k), targets, k, seen),
        "repeat": rank_metrics(repeat_ranking(past, users, k), targets, k, seen),
        "graph": rank_metrics(graph, targets, k, seen),
        "candidates (merged order)": rank_metrics(cands, targets, k, seen),
    }
    n_c = cfg.candidates.n_candidates
    results["candidate coverage"] = {
        f"recall@{n_c}": rank_metrics(cands, targets, n_c)[f"recall@{n_c}"]
    }

    if run.path("reranker").exists():
        from hybridrec.llm.train_reranker import make_samples, recent_history, rerank

        history = recent_history(past, item_text, cfg.prompt.max_history)
        samples, kept, _ = make_samples(users, cands, history, item_text, indexer, None, cfg.prompt)
        model = _build_reranker(run, vecs["users"].shape[1], [s.prompt for s in samples])
        model.load_trained(run.path("reranker"))
        ranked = rerank(model, samples, kept, vecs["users"].to(run.device), vecs["items"].to(run.device),
                        batch_size=cfg.rerank_train.batch_size)
        results["graph + LLM reranker"] = rank_metrics(ranked, targets, k, seen)

    run.path("metrics.json").write_text(json.dumps(results, indent=2))
    log(f"\ntest results (k={k}, {len(users):,} users)")
    for name, m in results.items():
        log(f"  {name:28s} " + "  ".join(f"{key}={val:.4f}" for key, val in m.items() if key != "n_users"))
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (defaults: synthetic data)")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    args = ap.parse_args(argv)

    run = Run(load_config(args.config))
    fns = {"prepare": stage_prepare, "vae": stage_vae, "graph": stage_graph,
           "rerank": stage_rerank, "evaluate": stage_evaluate}
    for name in STAGES:
        if name in args.stages:
            log(f"\n== {name} ==")
            fns[name](run)


if __name__ == "__main__":
    main()
