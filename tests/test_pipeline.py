"""End-to-end run on fake data with the tiny random LLM (about a minute on CPU)."""

import json

import yaml

from hybridrec.config import RunConfig, load_config
from hybridrec.pipeline import main


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("graph: {dimm: 3}")
    try:
        load_config(p)
    except KeyError as e:
        assert "dimm" in str(e)
    else:
        raise AssertionError("expected KeyError")
    assert isinstance(load_config(None), RunConfig)


def test_full_pipeline_smoke(tmp_path):
    cfg = {
        "run_dir": str(tmp_path / "run"),
        "train_device": "cpu",
        "tiny_llm": True,
        "data": {"source": "synthetic", "max_users": 120},
        "vae": {"epochs": 3},
        "graph": {"dim": 8, "epochs": 4, "eval_every": 2, "batch_size": 512},
        "candidates": {"n_candidates": 6, "n_repeat": 3},
        "reranker": {"lora_r": 2, "lora_alpha": 4, "projector_hidden": 8},
        "rerank_train": {"epochs": 1, "batch_size": 8, "grad_accum": 1, "eval_every_steps": 5},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    main(["--config", str(path)])

    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    for name in ("popular", "repeat", "graph", "graph + LLM reranker"):
        assert 0.0 <= metrics[name]["recall@10"] <= 1.0
    assert (tmp_path / "run" / "reranker" / "projectors.pt").exists()

    # Stages can be rerun alone from saved outputs.
    main(["--config", str(path), "--stages", "evaluate"])
