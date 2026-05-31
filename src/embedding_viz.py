"""
embedding_viz.py
────────────────
Extracts learned embedding weights from RecSysModel, reduces them to 2-D
with t-SNE and / or UMAP, and saves publication-quality scatter plots that
are then logged to MLflow as artifacts.

Usage (called from train.py after training):
    from src.embedding_viz import log_embedding_visualizations
    log_embedding_visualizations(model, BASE_PATH, max_points=3_000)

Dependencies:
    pip install scikit-learn umap-learn matplotlib seaborn
    (umap-learn is optional — t-SNE works without it)
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")                           # no display needed on server
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mlflow
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# UMAP is optional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning(
        "umap-learn not installed — UMAP plots will be skipped. "
        "Install with: pip install umap-learn"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class EmbVizConfig(NamedTuple):
    """Specification for a single embedding visualisation."""
    name: str           # must match a key in model.embedding_tables
    cmap: str           # matplotlib colourmap name
    description: str    # human-readable label for plot titles


# FIX 5: NamedTuple instead of raw tuples — positional access replaced by
# named attributes (cfg.name, cfg.cmap, cfg.description).
VIZ_CONFIG: list[EmbVizConfig] = [
    EmbVizConfig("Group4Code",           "tab20",   "Product sub-sub-group"),
    EmbVizConfig("Group3Code",           "tab20",   "Product sub-group"),
    EmbVizConfig("Group2Code",           "tab20",   "Product group"),
    EmbVizConfig("Group1Code",           "tab10",   "Product main group"),
    EmbVizConfig("FlavorID",             "tab20",   "Flavor"),
    EmbVizConfig("CustomerMainGroupID",  "tab10",   "Customer main group"),
    EmbVizConfig("CustomerSubGroupID",   "tab20",   "Customer sub-group"),
    EmbVizConfig("OwnerTypeID",          "tab10",   "Owner type"),
    EmbVizConfig("CustomerTypeID",       "tab10",   "Customer type"),
    EmbVizConfig("DefaultPaymentTypeID", "tab10",   "Payment type"),
    EmbVizConfig("StateCode",            "tab20",   "State"),
    EmbVizConfig("BranchCode",           "tab10",   "Branch"),
    EmbVizConfig("ZoneCode",             "viridis", "Zone"),
    EmbVizConfig("CityCode",             "viridis", "City"),
    EmbVizConfig("PathCode",             "plasma",  "Sales path"),
    EmbVizConfig("VisitorEmployeeID",    "viridis", "Visitor employee"),
    EmbVizConfig("PersonID",             "plasma",  "Customer (PersonID)"),
]

# Sequential colourmaps that warrant a colourbar rather than a legend.
_SEQUENTIAL_CMAPS = {"viridis", "plasma", "inferno", "magma", "cividis"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_weights(model: nn.Module, name: str) -> np.ndarray | None:
    """
    Pull the raw embedding weight matrix from model.embedding_tables.

    Returns shape (vocab_size - 1, embed_dim) as float32 numpy array,
    with row 0 (padding_idx) excluded so it does not anchor the projection.
    Returns None if the embedding is not found.
    """
    emb_tables = getattr(model, "embedding_tables", None)
    if emb_tables is None or name not in emb_tables:
        logger.warning(f"Embedding '{name}' not found in model — skipping.")
        return None

    # FIX 3 (partial): weights extracted inside torch.no_grad for safety,
    # even though .detach() already breaks the graph.
    with torch.no_grad():
        weights = emb_tables[name].weight.detach().cpu().float().numpy()
    return weights[1:]          # exclude padding row


def _subsample(weights: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample up to max_points rows when vocab exceeds the limit.
    Returns (sampled_weights, sampled_indices) where indices are the
    original vocabulary positions — used for consistent colour coding.
    """
    n = weights.shape[0]
    if n <= max_points:
        return weights, np.arange(n)
    idx = np.random.default_rng(42).choice(n, size=max_points, replace=False)
    return weights[idx], idx


# FIX 4: StandardScaler extracted into a shared helper so it is not
# duplicated inside both _reduce_tsne and _reduce_umap.
def _scale(weights: np.ndarray, dtype: type = np.float32) -> np.ndarray:
    """Standard-scale weights and cast to dtype."""
    return StandardScaler().fit_transform(weights).astype(dtype)


def _reduce_tsne(weights: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    """Reduce to 2-D with t-SNE."""
    scaled = _scale(weights)
    perplexity = min(perplexity, weights.shape[0] - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1_000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(scaled)


def _reduce_umap(weights: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """
    Reduce to 2-D with UMAP.

    Handles the sklearn ≥ 1.6 / older umap-learn version mismatch where
    force_all_finite was renamed to ensure_all_finite.  If both attempts
    fail, falls back to t-SNE with a warning.
    """
    scaled = _scale(weights)
    n_neighbors_safe = min(n_neighbors, weights.shape[0] - 1)
    kwargs = dict(n_components=2, n_neighbors=n_neighbors_safe,
                  min_dist=0.1, random_state=42)

    for attempt, arr in enumerate([scaled, np.ascontiguousarray(scaled, dtype=np.float64)]):
        try:
            return umap.UMAP(**kwargs).fit_transform(arr)
        except TypeError as exc:
            if "ensure_all_finite" not in str(exc) and "force_all_finite" not in str(exc):
                raise
            if attempt == 0:
                logger.warning(
                    f"UMAP version mismatch ({exc}). "
                    "Fix: pip install --upgrade umap-learn scikit-learn  "
                    "Retrying with float64 array …"
                )
            else:
                logger.warning("UMAP still failing — falling back to t-SNE for this embedding.")
                return _reduce_tsne(weights)

    return _reduce_tsne(weights)            # unreachable, satisfies type checker


def _get_cmap(name: str, n: int):
    """
    FIX 1: cm.get_cmap() was deprecated in matplotlib 3.7 and removed in 3.9.
    Use matplotlib.colormaps[] instead, which works from 3.5 onward.
    """
    try:
        # matplotlib ≥ 3.5 — preferred API
        return matplotlib.colormaps[name].resampled(n)
    except AttributeError:
        # fallback for very old matplotlib installations
        return cm.get_cmap(name, n)


def _make_plot(
    coords: np.ndarray,
    indices: np.ndarray,
    title: str,
    cmap_name: str,
    out_path: Path,
) -> Path:
    """
    Render one scatter plot with index-based colouring and save as PNG.

    Parameters
    ----------
    coords    : (N, 2) 2-D projection
    indices   : (N,)  original vocabulary indices (used for colour)
    title     : plot title
    cmap_name : matplotlib colourmap name
    out_path  : where to save the PNG
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    n_unique = len(np.unique(indices))
    cmap     = _get_cmap(cmap_name, n_unique)           # FIX 1: new API
    colors   = cmap(indices % n_unique)

    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors,
        s=max(1, min(20, 5_000 // len(coords))),        # auto-scale point size
        alpha=0.7,
        linewidths=0,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Dim 1", fontsize=10)
    ax.set_ylabel("Dim 2", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if cmap_name in _SEQUENTIAL_CMAPS:
        # Colourbar for sequential (large-vocab) embeddings
        sm = plt.cm.ScalarMappable(
            cmap=_get_cmap(cmap_name, n_unique),
            norm=plt.Normalize(indices.min(), indices.max()),
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Index", fraction=0.03, pad=0.04)
    else:
        # Legend for categorical (small-vocab) embeddings: show at most 20 labels
        n_show  = min(n_unique, 20)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cmap(i / n_unique), markersize=6)
            for i in range(n_show)
        ]
        labels = [str(i + 1) for i in range(n_show)]   # 1-indexed to match data
        if n_unique > n_show:
            labels[-1] = f"…{n_unique}"
        ax.legend(handles, labels, title="Index", fontsize=7,
                  title_fontsize=8, ncol=2, loc="upper right",
                  framealpha=0.6, markerscale=1.2)

    sns.despine(fig=fig, left=True, bottom=True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def log_embedding_visualizations(
    model: nn.Module,
    base_path: Path,
    max_points: int = 3_000,
    use_tsne: bool = True,
    use_umap: bool = True,
    mlflow_artifact_dir: str = "embeddings",
) -> None:
    """
    For each embedding in VIZ_CONFIG:
      1. Extract weights from model.embedding_tables
      2. Subsample to max_points if vocab is large
      3. Run t-SNE and/or UMAP projection
      4. Save PNG to  base_path/logs/embeddings/<method>/<name>.png
      5. Log all PNGs to MLflow under artifacts/<mlflow_artifact_dir>/

    Must be called inside an active ``mlflow.start_run()`` context.

    Parameters
    ----------
    model               : trained RecSysModel
    base_path           : project root (used to build output paths)
    max_points          : max vocabulary rows to plot (sub-sampled if larger)
    use_tsne            : whether to produce t-SNE plots
    use_umap            : whether to produce UMAP plots (requires umap-learn)
    mlflow_artifact_dir : MLflow artifact subfolder name
    """
    if use_umap and not UMAP_AVAILABLE:
        logger.warning("UMAP requested but umap-learn is not installed — skipping.")
        use_umap = False

    if not use_tsne and not use_umap:
        logger.warning("Both t-SNE and UMAP are disabled — nothing to visualise.")
        return

    # FIX 3: preserve and restore the model's training state so this
    # function is side-effect free with respect to train/eval mode.
    was_training = model.training
    model.eval()

    saved_paths: list[tuple[str, Path]] = []

    try:
        for cfg in VIZ_CONFIG:
            weights = _extract_weights(model, cfg.name)
            if weights is None:
                continue

            vocab_size, embed_dim = weights.shape
            logger.info(
                f"  Visualising '{cfg.name}' "
                f"(vocab={vocab_size}, embed_dim={embed_dim}) …"
            )

            sampled, indices = _subsample(weights, max_points)

            methods: list[tuple[str, np.ndarray]] = []
            if use_tsne:
                methods.append(("tsne", _reduce_tsne(sampled)))
            if use_umap:
                methods.append(("umap", _reduce_umap(sampled)))

            for method_name, coords in methods:
                out_path = (
                    base_path / "logs" / "embeddings" / method_name
                    / f"{cfg.name}.png"
                )
                title = (
                    f"{cfg.description}  [{cfg.name}]\n"
                    f"{method_name.upper()}  —  "
                    f"{len(sampled)}/{vocab_size} points  |  "
                    f"dim={embed_dim}"
                )
                _make_plot(coords, indices, title, cfg.cmap, out_path)
                saved_paths.append((method_name, out_path))
                logger.info(f"    ✓ {method_name.upper()} → {out_path.name}")

    finally:
        # Always restore training state — even if an exception is raised.
        model.train(was_training)

    # Log everything to MLflow in one pass
    for method_name, path in saved_paths:
        mlflow.log_artifact(
            str(path),
            artifact_path=f"{mlflow_artifact_dir}/{method_name}",
        )

    logger.info(
        f"Embedding visualizations logged to MLflow "
        f"under '{mlflow_artifact_dir}/' "
        f"({len(saved_paths)} plots total)."
    )