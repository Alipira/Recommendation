import time
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, IterableDataset
from pathlib import Path
from loguru import logger
from matplotlib import pyplot as plt
from typing import Any

from src.recsys_model import get_recsys_model
from utils.read_env import read_redis_env_vars
from src.consumer import data_gen, ConsumerError
from src.embedding_viz import log_embedding_visualizations
from src.training_viz import TrainingVisualizer


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = Path(__file__).parent.parent

EPOCHS = 10
STEPS_PER_EPOCH = 300 #  278550
STEPS_PER_EPOCH_VAL = 150 #  4500
STEPS_PER_EPOCH_TEST = 25

INTERACTION_DIM = 2
SHOPPING_CART_INTERACTION_DIM = 2
RESIDUAL_EMBEDDINGS_TO_NECK = True
RESIDUAL_SCC_DIGEST_TO_NECK = True

INITIAL_LR = 1e-3
LR_DECAY_STEPS = 100_000
LR_DECAY_RATE = 0.85
LR_MIN = 1e-6

WEIGHT_DECAY = 1e-3
GLOBAL_CLIP_NORM = 1.0
DROPOUT = 0.15
EARLY_STOP_PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "RecSys_Model_weighted_huber_loss"
MLFLOW_RUN_NAME = "test combined weighted Hubber loss"

# CHANGED: 0.3 keeps more of the rare-sample error in the quadratic
# (high-gradient) region. On log1p-scaled targets (range ≈ 0–6),
# delta=1.0 pushes almost all rare-sample loss into the flat linear
# region, giving the model weak gradient signal exactly where it needs it most.
HUBER_DELTA = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Loss: combined frequency + value weighting
# ─────────────────────────────────────────────────────────────────────────────

def combined_weighted_huber_loss(
    preds: torch.Tensor,   # (B, 1)
    target: torch.Tensor,   # (B, 1)— already log1p(Qty)
    freq_weights: torch.Tensor,   # (B,)— inverse-sqrt frequency, pre-normalised
) -> torch.Tensor:
    """
    Huber loss weighted by BOTH sample rarity AND purchase magnitude.

    Weight decomposition
    ────────────────────
    • freq_weight  : 1/sqrt(count(personId, group4code))
                     — rare customer-product pairs get higher gradient.
    • value_weight : target + 1.0
                     — target is already log1p(Qty), so this is log1p(Qty)+1,
                     which gives proportionally more weight to high-quantity
                     samples WITHOUT double-applying log (no expm1 needed).

    Both components are independently normalised so neither dominates,
    then multiplied and re-normalised so the overall loss scale is stable
    and the optimiser LR does not need retuning.
    """
    #breakpoint()
    preds_flat = preds  # .squeeze(-1)   # (B,)
    target_flat = target  # .squeeze(-1)  # (B,)

    # ── Value weight (log-space magnitude) ──────────────────────────────────
    # target_flat is log1p(Qty); +1 avoids zero-weighting for Qty=0 samples.
    value_w = target_flat + 1.0
    value_w = value_w / value_w.mean().clamp(min=1e-8)

    # ── Frequency weight (rarity) ───────────────────────────────────────────
    # freq_weights arrive already normalised from data_generator (mean ≈ 1),
    # but re-normalise per batch in case the batch distribution shifts.
    freq_w = freq_weights / freq_weights.mean().clamp(min=1e-8)

    # ── Combined weight ──────────────────────────────────────────────────────
    combined = value_w * freq_w
    combined = combined / combined.mean().clamp(min=1e-8)  # final normalisation

    element_loss = F.huber_loss(
        preds_flat, target_flat,
        delta=HUBER_DELTA,
        reduction="none",
    )
    return (element_loss * combined).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RedisStreamDataset(IterableDataset):
    def __init__(self, num_epochs: int, stream_key: str, steps: int):
        self.num_epochs = num_epochs
        self.stream_key = stream_key
        self.steps = steps

    def __iter__(self):
        logger.info(f"Worker info: {torch.utils.data.get_worker_info()}")
        yield from data_gen(self.num_epochs, self.stream_key, self.steps)


# ─────────────────────────────────────────────────────────────────────────────
# Device transfer — CHANGED: handles freq_weights (third element)
# ─────────────────────────────────────────────────────────────────────────────

def to_device(
    inputs:       dict[str, Any],
    target:       Any,
    freq_weights: Any,
    device:       torch.device,
) -> tuple:
    def _to_tensor(v, key):
        if not isinstance(v, torch.Tensor):
            dtype = torch.long if key in ("i1", "i3") else torch.float32
            v = torch.tensor(v, dtype=dtype)
        return v.to(device)

    inputs_t = {k: _to_tensor(v, k) for k, v in inputs.items()}

    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)
    target_t = target.to(device)

    # freq_weights: (B,) float32
    if not isinstance(freq_weights, torch.Tensor):
        freq_weights = torch.tensor(freq_weights, dtype=torch.float32)
    freq_weights_t = freq_weights.to(device)

    return inputs_t, target_t, freq_weights_t


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict[str, Any] = {}

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            logger.info(f"  ✓ val_loss improved → {val_loss:.6f}  checkpoint saved")
        else:
            self.counter += 1
            logger.info(f"  ✗ no improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                logger.info("Early stopping triggered.")
                return True
        return False

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state:
            model.load_state_dict(self.best_state)
            logger.info("Best weights restored.")


# ─────────────────────────────────────────────────────────────────────────────
# Training / Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:          nn.Module,
    loader:         DataLoader,
    device:         torch.device,
    optimizer:      torch.optim.Optimizer | None = None,
    step_scheduler: StepLR | None = None,
    visualizer:     TrainingVisualizer | None = None,
    max_steps:      int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = total_mae = total_mse = 0.0
    n_batches = 0

    desc = "TRAIN" if training else "VAL"
    pbar = tqdm(total=max_steps, desc=desc, unit="batch", dynamic_ncols=True)

    with torch.set_grad_enabled(training):
        # CHANGED: unpack 3-tuple from loader
        for step, (inputs, target, freq_weights) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break

            inputs, target, freq_weights = to_device(inputs, target, freq_weights, device)
            preds = model(inputs)   # (B, 1)

            # CHANGED: use combined_weighted_huber_loss
            loss = combined_weighted_huber_loss(preds, target, freq_weights)

            if training:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), GLOBAL_CLIP_NORM)
                optimizer.step()

                if step_scheduler is not None:
                    step_scheduler.step()

                if visualizer is not None:
                    visualizer.on_batch_end(loss.item())

            with torch.no_grad():
                total_loss += loss.item()
                total_mae += (preds - target).abs().mean().item()
                total_mse += ((preds - target) ** 2).mean().item()
                n_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss / n_batches:.4f}",
                "mae":  f"{total_mae  / n_batches:.4f}",
                "step": f"{n_batches}/{max_steps}",
            })
            pbar.update(1)

    pbar.close()
    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "mae":  total_mae / denom,
        "mse":  total_mse / denom,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, epoch: int, base_path: Path) -> Path:
    ckpt_dir = base_path / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = ckpt_dir / f"epoch_{epoch:02d}_{ts}.pt"
    torch.save(model.state_dict(), path)
    return path


def save_plot(history: dict[str, list], base_path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    ep = np.arange(1, len(history["train_loss"]) + 1)

    for ax, (tr_key, val_key), title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_mae", "val_mae"), ("train_mse", "val_mse")],
        ["Loss", "MAE", "MSE"],
    ):
        ax.plot(ep, history[tr_key],  "b-o", label=f"Train {title}")
        ax.plot(ep, history[val_key], "r-x", label=f"Val {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = base_path / "logs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"history_{datetime.now()}.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main (unchanged except MLflow param logging reflects new config)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    _, _, _, stream_key, stream_key_val, stream_key_test, _, _ = read_redis_env_vars()
    missing = {
        "stream_key": stream_key,
        "stream_key_val": stream_key_val,
        "stream_key_test": stream_key_test
    }
    none_keys = [k for k, v in missing.items() if not v]
    if none_keys:
        raise EnvironmentError(
            f"Stream keys missing from .env: {none_keys}"
        )

    model = get_recsys_model(
        interaction_dim=INTERACTION_DIM,
        shopping_cart_interaction_dim=SHOPPING_CART_INTERACTION_DIM,
        residual_embeddings_to_neck=RESIDUAL_EMBEDDINGS_TO_NECK,
        residual_SCC_digest_to_neck=RESIDUAL_SCC_DIGEST_TO_NECK,
        dropout=DROPOUT,
    ).to(DEVICE)
    logger.info(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    step_scheduler = StepLR(optimizer, step_size=LR_DECAY_STEPS, gamma=LR_DECAY_RATE)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=EARLY_STOP_PATIENCE, min_lr=LR_MIN,
    )
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    viz = TrainingVisualizer(base_path=BASE_PATH, device=DEVICE)

    loader_kwargs = dict(batch_size=None, num_workers=0,
                         pin_memory=(DEVICE.type == "cuda"))

    train_loader = DataLoader(
        RedisStreamDataset(EPOCHS, stream_key, STEPS_PER_EPOCH),
        **loader_kwargs
    )
    val_loader = DataLoader(
        RedisStreamDataset(1, stream_key_val, STEPS_PER_EPOCH_VAL),
        **loader_kwargs
    )
    test_loader = DataLoader(
        RedisStreamDataset(1, stream_key_test, STEPS_PER_EPOCH_TEST),
        **loader_kwargs
    )
    test_data = [batch for batch in test_loader]

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_mae":  [], "val_mae":  [],
        "train_mse":  [], "val_mse":  [],
    }

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):

        mlflow.log_params({
            "interaction_dim":               INTERACTION_DIM,
            "shopping_cart_interaction_dim": SHOPPING_CART_INTERACTION_DIM,
            "residual_embeddings_to_neck":   RESIDUAL_EMBEDDINGS_TO_NECK,
            "residual_scc_digest_to_neck":   RESIDUAL_SCC_DIGEST_TO_NECK,
            "initial_lr":                    INITIAL_LR,
            "lr_decay_steps":                LR_DECAY_STEPS,
            "lr_decay_rate":                 LR_DECAY_RATE,
            "weight_decay":                  WEIGHT_DECAY,
            "global_clip_norm":              GLOBAL_CLIP_NORM,
            "dropout":                       DROPOUT,
            "epochs":                        EPOCHS,
            "steps_per_epoch":               STEPS_PER_EPOCH,
            "early_stop_patience":           EARLY_STOP_PATIENCE,
            "device":                        str(DEVICE),
            "total_params":                  total_params,
            "trainable_params":              trainable_params,
            "neck_dim":                      model.neck_dim,
            "scc_digest_dim":                model.scc_digest_dim,
            "loss_fn":                       "CombinedWeightedHuberLoss",
            "huber_delta":                   HUBER_DELTA,
            "loss_weight_scheme":            "freq(inv-sqrt) * value(log1p+1) normalised",
        })

        logger.info("Training started …")
        stopped_early = False

        try:
            for epoch in range(1, EPOCHS + 1):
                logger.info(f"\n── Epoch {epoch}/{EPOCHS} ──────────────────────")
                print(f"epoch: {EPOCHS}")
            
                train_metrics = run_epoch(
                    model, train_loader, DEVICE,
                    optimizer=optimizer,
                    step_scheduler=step_scheduler,
                    visualizer=viz,
                    max_steps=STEPS_PER_EPOCH,
                )

                val_metrics = run_epoch(
                    model, val_loader, DEVICE,
                    max_steps=STEPS_PER_EPOCH_VAL,
                )
                plateau_scheduler.step(val_metrics["loss"])

                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  train → loss={train_metrics['loss']:.6f}  "
                    f"mae={train_metrics['mae']:.6f}  mse={train_metrics['mse']:.6f}"
                )
                logger.info(
                    f"  val   → loss={val_metrics['loss']:.6f}  "
                    f"mae={val_metrics['mae']:.6f}  mse={val_metrics['mse']:.6f}  "
                    f"lr={lr:.8f}"
                )

                for split, m in [("train", train_metrics), ("val", val_metrics)]:
                    for k, v in m.items():
                        history[f"{split}_{k}"].append(v)

                viz.on_epoch_end(epoch, train_metrics, val_metrics)
                logger.info("Generating prediction diagnostic plots …")
                viz.visualize_predictions(
                    model,
                    test_data,
                    epoch,
                    STEPS_PER_EPOCH_TEST,
                )
                logger.info("Befor log metric in mlflow")    
                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "train_mae":  train_metrics["mae"],
                        "train_mse":  train_metrics["mse"],
                        "val_loss":   val_metrics["loss"],
                        "val_mae":    val_metrics["mae"],
                        "val_mse":    val_metrics["mse"],
                        "lr":         lr,
                    },
                    step=epoch,
                )

                ckpt_path = save_checkpoint(model, epoch, BASE_PATH)
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")

                if early_stopping.step(val_metrics["loss"], model):
                    stopped_early = True
                    break

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt — saving current state.")
            mlflow.set_tag("stopped_reason", "KeyboardInterrupt")

        except ConsumerError as exc:
            logger.error(f"Consumer error: {exc}")
            mlflow.set_tag("stopped_reason", f"ConsumerError: {exc}")
        except Exceptions as exc:
            logger.error(f"Training error: {exc}")
            mlflow.set_tag("stopped_reason", f"Exception: {exc}")

        finally:
            early_stopping.restore_best(model)

            if stopped_early:
                mlflow.set_tag("stopped_reason", "EarlyStopping")

            if history["val_loss"]:
                logger.info("Running test evaluation …")
                try:
                    test_metrics = run_epoch(
                        model,
                        test_data,
                        DEVICE,
                        max_steps=STEPS_PER_EPOCH_TEST,
                    )
                    logger.info(
                        f"  test  → loss={test_metrics['loss']:.6f}  "
                        f"mae={test_metrics['mae']:.6f}  mse={test_metrics['mse']:.6f}"
                    )
                    mlflow.log_metrics({
                        "test_loss": test_metrics["loss"],
                        "test_mae":  test_metrics["mae"],
                        "test_mse":  test_metrics["mse"],
                    })
                    logger.info("Generating prediction diagnostic plots …")
                    viz.visualize_predictions(
                        model,
                        test_loader,
                        epoch,
                        max_batches=STEPS_PER_EPOCH_TEST,
                    )
                except ConsumerError as exc:
                    logger.error(f"Test evaluation skipped — consumer error: {exc}")

            mlflow.pytorch.log_model(
                model,
                name="model",
                registered_model_name="RecSysModel",
            )

            if history["train_loss"]:
                plot_path = save_plot(history, BASE_PATH)
                mlflow.log_artifact(str(plot_path), artifact_path="plots")

            logger.info("Generating embedding visualizations …")
            log_embedding_visualizations(
                model=model,
                base_path=BASE_PATH,
                max_points=3_000,
                use_tsne=True,
                use_umap=True,
            )

            logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
            logger.info("Training session ended.")


if __name__ == "__main__":
    main()
