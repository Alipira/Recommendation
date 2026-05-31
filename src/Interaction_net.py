"""
interaction_net.py
──────────────────
Pairwise field-interaction network for recommendation systems.

Supports both:
  • 2D input  (B, D)      — standard interaction path
  • 3D input  (B, T, D)   — shopping-cart / sequence path

For every pair (i, j) the block computes:
    concat(field_i, field_j)
        → Linear(hidden_dim, bias=False)
        → BatchNorm1d
        → LeakyReLU
        → Dropout
        → Linear(interaction_dim)
        → LeakyReLU

All pair outputs are concatenated along the feature dimension.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import torch
import torch.nn as nn


class InteractionNet(nn.Module):
    """Pairwise feature-interaction network.

    Args:
        field_names:     Human-readable label per field.
        field_dims:      Feature dimensionality of each field.
        interaction_dim: Output size per pair block.
        dropout:         Dropout probability inside each pair block.
        negative_slope:  LeakyReLU negative slope.
    """

    def __init__(
        self,
        field_names: list[str],
        field_dims: list[int],
        interaction_dim: int,
        dropout: float = 0.15,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        if len(field_names) != len(field_dims):
            raise ValueError(
                f"`field_names` and `field_dims` must have the same length, "
                f"got {len(field_names)} vs {len(field_dims)}."
            )
        if len(field_names) < 2:
            raise ValueError("At least two fields are required to compute interactions.")

        self.field_names     = list(field_names)
        self.field_dims      = list(field_dims)
        self.interaction_dim = interaction_dim
        self.dropout_p       = dropout
        self.negative_slope  = negative_slope

        # All (i, j) pairs — order is fixed and shared by __init__ and forward.
        self.field_pairs: list[tuple[int, int]] = list(
            combinations(range(len(field_names)), 2)
        )

        # One nn.Sequential per pair keeps each block self-contained.
        # nn.ModuleList ensures every block's parameters are registered so
        # that .parameters(), .state_dict(), and .to(device) all work correctly.
        self.pair_blocks = nn.ModuleList()
        for i, j in self.field_pairs:
            in_dim     = field_dims[i] + field_dims[j]
            hidden_dim = (in_dim + interaction_dim) // 2

            block = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, interaction_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
            )
            self._init_block(block)
            self.pair_blocks.append(block)

    # ── Initialisation ───────────────────────────────────────────────────────

    @staticmethod
    def _init_block(block: nn.Sequential) -> None:
        for layer in block:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, fields: list[torch.Tensor]) -> torch.Tensor:
        """Compute all pairwise interactions and concatenate.

        Args:
            fields: One tensor per field.
                • 2D mode — shape ``(B, D_k)``
                • 3D mode — shape ``(B, T, D_k)``  (sequence / shopping-cart path)

        Returns:
            • 2D input → ``(B, num_pairs × interaction_dim)``
            • 3D input → ``(B, T, num_pairs × interaction_dim)``
        """
        if len(fields) != len(self.field_names):
            raise ValueError(
                f"Expected {len(self.field_names)} field tensors, got {len(fields)}."
            )

        # 3D sequence mode: flatten (B, T, D) → (B·T, D) for BatchNorm1d,
        # then restore the time dimension on the way out.
        is_seq = fields[0].dim() == 3
        if is_seq:
            B, T, _ = fields[0].shape
            flat = [f.reshape(B * T, -1) for f in fields]
        else:
            flat = fields

        interactions: list[torch.Tensor] = [
            block(torch.cat([flat[i], flat[j]], dim=-1))
            for block, (i, j) in zip(self.pair_blocks, self.field_pairs)
        ]
        out = torch.cat(interactions, dim=-1)

        return out.reshape(B, T, -1) if is_seq else out

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def num_pairs(self) -> int:
        """Number of pairwise interaction blocks."""
        return len(self.field_pairs)

    @property
    def output_dim(self) -> int:
        """Total output width: ``num_pairs × interaction_dim``."""
        return self.num_pairs * self.interaction_dim

    # ── Serialisation ────────────────────────────────────────────────────────

    def get_config(self) -> dict[str, Any]:
        """Return constructor arguments for checkpoint / reconstruction."""
        return {
            "field_names":     self.field_names,
            "field_dims":      self.field_dims,
            "interaction_dim": self.interaction_dim,
            "dropout":         self.dropout_p,
            "negative_slope":  self.negative_slope,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> InteractionNet:
        """Reconstruct from a config dict produced by ``get_config``."""
        return cls(**config)

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"({self.field_names[i]}×{self.field_names[j]})"
            for i, j in self.field_pairs
        )
        return (
            f"{self.__class__.__name__}("
            f"fields={self.field_names}, "
            f"interaction_dim={self.interaction_dim}, "
            f"num_pairs={self.num_pairs}, "
            f"output_dim={self.output_dim})"
        )
