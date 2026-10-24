"""Run configuration, loaded from a YAML file. Missing keys keep their defaults."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hybridrec.llm.prompt import PromptConfig
from hybridrec.llm.reranker import RerankerConfig
from hybridrec.llm.train_reranker import RerankTrainConfig
from hybridrec.models.lightgcn import GraphConfig
from hybridrec.models.vae import VAEConfig


@dataclass
class DataConfig:
    source: str = "synthetic"         # synthetic | instacart | erp
    path: str = ""                    # instacart folder
    max_users: int | None = None      # sample users for quicker runs
    min_baskets: int = 3
    category_col: str | None = None   # item column used for user category shares
    user_vectors: str | None = None   # parquet of ready-made customer vectors (e.g. CustomerVAE)
    erp_connection_env: str = "ERP_DB_URL"  # env var holding the SQLAlchemy URL
    seed: int = 0


@dataclass
class CandidateConfig:
    n_candidates: int = 20
    n_repeat: int = 8


@dataclass
class RunConfig:
    run_dir: str = "runs/default"
    train_device: str = "cuda:1"      # falls back to CPU when CUDA is missing
    eval_k: int = 10
    use_vae: bool = True
    tiny_llm: bool = False            # offline random model, for smoke tests only
    reranker_valid_share: float = 0.1  # share of validation users kept for reranker early stopping
    rerank_max_train_users: int | None = None
    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    rerank_train: RerankTrainConfig = field(default_factory=RerankTrainConfig)


def _fill(cls, values: dict | None):
    obj = cls()
    if not values:
        return obj
    names = {f.name: f for f in dataclasses.fields(cls)}
    for key, val in values.items():
        if key not in names:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(val, dict):
            val = _fill(type(current), val)
        setattr(obj, key, val)
    return obj


def load_config(path: str | Path | None) -> RunConfig:
    if path is None:
        return RunConfig()
    return _fill(RunConfig, yaml.safe_load(Path(path).read_text()) or {})


def config_to_dict(cfg: RunConfig) -> dict:
    return dataclasses.asdict(cfg)
