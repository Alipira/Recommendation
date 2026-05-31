"""
training_visualizer.py
──────────────────────
PyTorch port of the original TF SaveModelAndVisualize Keras callback.

In PyTorch there is no built-in callback system.  This module provides a
``TrainingVisualizer`` class whose methods are called explicitly from the
training loop in train.py:

    viz = TrainingVisualizer(base_path=BASE_PATH, device=DEVICE)

    # inside run_epoch (per batch):
    viz.on_batch_end(loss=loss.item())

    # inside main() (per epoch):
    viz.on_epoch_end(epoch, train_metrics, val_metrics)

    # after training:
    viz.visualize_predictions(model, test_loader, loss_fn, max_batches=25)
"""
from __future__ import annotations

import time
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import matplotlib

from pathlib import Path
from loguru import logger
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

_EPS = 1e-6
matplotlib.use("Agg")


def _ts() -> str:
    """
    FIX 2: datetime.now() produced filenames with colons (invalid on Windows
    and ugly on Linux).  time.strftime produces clean, filesystem-safe strings.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _save(fig: plt.Figure, path: Path) -> None:
    """Ensure output directory exists, save the figure, close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TrainingVisualizer:
    """
    Collects per-batch and per-epoch metrics and produces diagnostic plots.

    All plots are saved to disk under ``base_path/logs/results/`` and
    optionally logged to the active MLflow run.

    Parameters
    ----------
    base_path       : project root Path object
    device          : torch.device used for inference during visualize_predictions
    log_to_mlflow   : whether to log generated plots to MLflow as artifacts
    roc_threshold   : value above which the target is treated as positive
                      for the ROC curve (regression → binary classification)
    """

    def __init__(
        self,
        base_path: Path,
        device: torch.device,
        log_to_mlflow: bool = True,
        roc_threshold: float = 0.1,
    ) -> None:
        self.base_path = base_path
        self.device = device
        self.log_to_mlflow = log_to_mlflow
        self.roc_threshold = roc_threshold

        # Accumulated history — mirrors the original TF callback's state
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self.batch_losses: list[float] = []

        self._out = base_path / "logs" / "results"

    # ── Per-batch hook ───────────────────────────────────────────────────────

    def on_batch_end(self, loss: float) -> None:
        """Call after every training optimizer step."""
        self.batch_losses.append(loss)

    # ── Per-epoch hook ───────────────────────────────────────────────────────

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """
        Call at the end of every epoch.  Records metrics and saves the loss
        plot.  Prediction plots are generated separately via
        ``visualize_predictions`` to avoid running the full test set every epoch.

        Parameters
        ----------
        epoch         : 1-based epoch number
        train_metrics : dict with at least the key "loss"
        val_metrics   : dict with at least the key "loss"
        """
        self.train_losses.append(train_metrics["loss"])
        self.val_losses.append(val_metrics["loss"])
        self._plot_loss(epoch)

    # ── End-of-training prediction visualisation ─────────────────────────────

    def visualize_predictions(
        self,
        model: nn.Module,
        loader: DataLoader,
        epoch: int,
        max_batches: int | None = None,
    ) -> None:
        """
        Run the model on ``loader``, then produce:
          • scatter plot  — predictions vs ground truth (linear + log scale)
          • 2-D histogram — predictions vs ground truth (log-density)
          • ROC curve     — binary classification view at roc_threshold
        """
        y_true, y_pred = self._collect_predictions(model, loader, max_batches)
        ts = _ts()
        paths: list[Path] = [
            self._plot_pred_vs_true(epoch, y_true, y_pred, ts),
            # self._plot_hist2d(epoch, y_true, y_pred, ts),
            self._plot_roc(epoch, y_true, y_pred, ts),
        ]
        if self.log_to_mlflow:
            for p in paths:
                mlflow.log_artifact(str(p), artifact_path="prediction_plots")


    def _collect_predictions(
        self,
        model: nn.Module,
        loader: DataLoader,
        max_batches: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run model in eval mode and collect predictions and labels."""
        was_training = model.training
        model.eval()

        all_labels: list[np.ndarray] = []
        all_preds:  list[np.ndarray] = []

        try:
            with torch.no_grad():
                for i, (inputs, target) in enumerate(loader):
                    if max_batches is not None and i >= max_batches:
                        break

                    # Move inputs to device (mirrors to_device in train.py)
                    inputs = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor)
                        else torch.tensor(v).to(self.device)
                        for k, v in inputs.items()
                    }
                    preds = model(inputs).cpu().numpy()  # squeeze(-1) was omited by ALI
                    all_preds.append(preds)

                    if isinstance(target, torch.Tensor):
                        all_labels.append(target.cpu().numpy())
                    else:
                        all_labels.append(np.asarray(target))

        finally:
            model.train(was_training)

        return np.concatenate(all_labels), np.concatenate(all_preds)

    # ── Private: plot helpers ────────────────────────────────────────────────

    def _plot_loss(self, epoch: int) -> Path:
        """
        Loss curve with per-batch noise in the background.

        FIX 4: the original code assumed batch_losses span exactly one epoch,
        so the x-axis used `epoch / len(batch_losses)` as a scale factor —
        correct for epoch 1 only.  Now we map batches to their true global
        x-position by computing cumulative batches per epoch.
        """
        fig, ax = plt.subplots(figsize=(16, 5), layout="constrained")

        if self.batch_losses:
            n_epochs_so_far = len(self.train_losses)
            batches_per_epoch = len(self.batch_losses) / max(n_epochs_so_far, 1)
            # FIX 4: each batch index maps to a fractional epoch position
            x_batches = (np.arange(len(self.batch_losses)) + 1) / max(batches_per_epoch, 1)
            ax.plot(
                x_batches, self.batch_losses,
                color=(177/255, 194/255, 243/255),
                alpha=0.6, linewidth=0.8,
                label="Training batch loss",
            )

        ep = np.arange(len(self.train_losses))
        ax.plot(ep + 0.5, self.train_losses, "g+-", label="Training loss", alpha=0.85)
        ax.plot(ep + 1,   self.val_losses,   "rx-", label="Validation loss", alpha=0.85)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

        path = self._out / f"history_epoch_{epoch:02d}_{_ts()}.png"
        _save(fig, path)

        if self.log_to_mlflow:
            mlflow.log_artifact(str(path), artifact_path="loss_plots")

        return path

    def _plot_pred_vs_true(
        self, epoch: int, y_true: np.ndarray, y_pred: np.ndarray, ts: str
    ) -> Path:
        """Scatter: predictions vs ground truth on linear and log scales."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        for ax, (yt, yp, label) in zip(
            axes,
            [
                (y_true, y_pred, "Linear scale"),
                (np.log10(y_true + _EPS), np.log10(y_pred + _EPS), "Log₁₀ scale"),
            ],
        ):
            ax.scatter(yt, yp, alpha=0.5, edgecolors="none", s=4)

            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")

            ax.set_xlabel("Ground truth")
            ax.set_ylabel("Prediction")
            ax.set_title(label)
            ax.set_xlim(lo - 0.01, hi + 0.01)
            ax.set_ylim(lo - 0.01, hi + 0.01)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)

        path = self._out / f"pred_vs_true_epoch_{epoch:02d}_{ts}.png"
        _save(fig, path)
        # logger.info(f"  ✓ prediction scatter → {path.name}")
        return path

    def _plot_hist2d(
        self,
        epoch: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ts: str,
        bins: int = 200,
    ) -> Path:
        """2-D log-density histogram of predictions vs ground truth."""
        fig, ax = plt.subplots(figsize=(8, 8))
        logger.info("1111111111111111111")
        print(f"Axis inside the _plot_hist2D {ax}")
        h = ax.hist2d(y_true, y_pred, bins=bins, norm=LogNorm())
        plt.colorbar(h[3], ax=ax, label="Count (log scale)")
        logger.info("22222222222222222222222")

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        logger.info("333333333333333333333")

        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
        ax.set_xlabel("True values")
        ax.set_ylabel("Predictions")
        ax.set_title("True vs Predicted (2-D histogram)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        logger.info("44444444444444444444444")

        path = self._out / f"pred_hist2d_epoch_{epoch:02d}_{ts}.png"
        logger.info("555555555555555555")

        _save(fig, path)
        logger.info("666666666")

        # logger.info(f"  ✓ 2D histogram → {path.name}")
        return path

    def _plot_roc(
        self,
        epoch: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ts: str,
    ) -> Path:
        """ROC curve treating values ≥ roc_threshold as positive."""
        y_bin = (y_true >= self.roc_threshold).astype(int)

        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            logger.warning(
                f"ROC curve skipped: all labels are the same class at "
                f"threshold={self.roc_threshold}. "
                f"Adjust roc_threshold in TrainingVisualizer."
            )
            # Return a dummy path so the caller does not need to handle None
            return self._out / f"roc_skipped_epoch_{epoch:02d}_{ts}.txt"

        fpr, tpr, _ = roc_curve(y_bin, y_pred.astype(float))
        auc = roc_auc_score(y_bin, y_pred.astype(float))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC curve  (threshold = {self.roc_threshold})")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)

        path = self._out / f"roc_curve_epoch_{epoch:02d}_{ts}.png"
        _save(fig, path)
        # logger.info(f"  ✓ ROC curve (AUC={auc:.3f}) → {path.name}")
        return path
