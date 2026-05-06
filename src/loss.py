"""
loss.py
───────
Custom loss functions for the RecSys training pipeline.

Both losses are mean-reduced over the batch, matching PyTorch convention
(torch built-ins like nn.MSELoss default to mean reduction).

Why these losses for recommendation?
─────────────────────────────────────
• L2 (MSE) alone penalises large errors heavily but ignores the relative
  scale of predictions — a 10-unit error on a target of 1000 is treated
  the same as on a target of 2.

• Adding L1 (MAE) makes the loss robust to outliers and stabilises
  training when occasional very large targets appear.

• Adding a log-space term (l2_plus_log_l2) further compresses the dynamic
  range so the model learns proportional errors rather than absolute ones —
  critical when targets span several orders of magnitude (e.g. sales qty).
"""

import torch
import torch.nn.functional as F


# ── Epsilon for numerical stability in log terms ──────────────────────────────
_EPS = 1e-4


def l2_plus_l1_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    MSE  +  MAE loss (element-wise mean over batch).

    L = mean[ (y - ŷ)²  +  |y - ŷ| ]

    Properties
    ----------
    • L2 term: punishes large deviations quadratically → fast convergence
      when predictions are far from targets.
    • L1 term: constant gradient near the solution → more robust to
      outliers, prevents the L2 term from over-correcting on noisy labels.

    Args
    ----
    y_pred : (B, 1)  model output
    y_true : (B, 1)  ground-truth labels
    """
    diff = y_pred - y_true
    return (diff.pow(2) + diff.abs()).mean()


def l2_plus_log_l2_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    MSE  +  log-space MSE loss (element-wise mean over batch).

    L = mean[ (y - ŷ)²  +  (log(y + ε) - log(ŷ + ε))² ]

    Properties
    ----------
    • The log-space term makes the loss scale-invariant: a relative error
      of 10 % costs the same whether the target is 1 or 10 000.
    • ε = 1e-4 keeps log() defined when predictions or targets are zero or
      slightly negative (e.g. after ReLU output with numerical noise).
    • Useful when sales quantities or amounts span multiple orders of
      magnitude within the same batch.

    Args
    ----
    y_pred : (B, 1)  model output  — should be ≥ 0 (ReLU head)
    y_true : (B, 1)  ground-truth labels  — should be ≥ 0
    """
    diff     = y_pred - y_true
    log_diff = torch.log(y_pred + _EPS) - torch.log(y_true + _EPS)
    return (diff.pow(2) + log_diff.pow(2)).mean()

