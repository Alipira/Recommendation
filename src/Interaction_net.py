"""
interaction_net.py
──────────────────
Pairwise field-interaction network for recommendation systems.

Supports both:
  • 2D input  (B, D)      — main interaction path
  • 3D input  (B, T, D)   — shopping-cart sequence path

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

import torch
import torch.nn as nn
from itertools import combinations
from typing import List


class InteractionNet(nn.Module):

    def __init__(
        self,
        field_names: List[str],
        field_dims: List[int],
        interaction_dim: int,
        dropout: float = 0.15,
        negative_slope: float = 0.01,
    ):
        """
        Args:l
            field_names:     Human-readable label per field.
            field_dims:      Feature dimensionality of each field.
            interaction_dim: Output size per pair block.
            dropout:         Dropout probability inside each pair block.
            negative_slope:  LeakyReLU negative slope.
        """
        super().__init__()

        assert len(field_names) == len(field_dims), (
            f"field_names and field_dims must match: "
            f"got {len(field_names)} vs {len(field_dims)}"
        )

        self.field_names     = list(field_names)
        self.field_dims      = list(field_dims)
        self.interaction_dim = interaction_dim
        self.dropout_p       = dropout
        self.negative_slope  = negative_slope

        # All (i, j) pairs — order fixed and shared between __init__ and forward
        self.field_pairs: List[tuple] = list(combinations(range(len(field_names)), 2))

        # Flat ModuleList: one nn.Sequential per pair so all params are registered
        self.pair_blocks = nn.ModuleList()
        for i, j in self.field_pairs:
            in_dim     = field_dims[i] + field_dims[j]
            hidden_dim = (in_dim + interaction_dim) // 2

            block = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, interaction_dim),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            )
            self._init_block(block)
            self.pair_blocks.append(block)

    # ── Init ─────────────────────────────────────────────────────────────────
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

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, fields: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            fields: One tensor per field.
              • 2D mode — shape (B, D_k)
              • 3D mode — shape (B, T, D_k)   [shopping-cart sequence path]

        Returns:
            • 2D → (B,    num_pairs * interaction_dim)
            • 3D → (B, T, num_pairs * interaction_dim)
        """
        assert len(fields) == len(self.field_names), (
            f"Expected {len(self.field_names)} fields, got {len(fields)}"
        )

        # 3D sequence mode: flatten (B, T, D) → (B·T, D) for BatchNorm1d
        is_seq = fields[0].dim() == 3
        if is_seq:
            B, T, _ = fields[0].shape
            flat = [f.reshape(B * T, -1) for f in fields]
        else:
            flat = fields

        interactions: List[torch.Tensor] = []
        for block, (i, j) in zip(self.pair_blocks, self.field_pairs):
            pair_input = torch.cat([flat[i], flat[j]], dim=-1)
            interactions.append(block(pair_input))

        out = torch.cat(interactions, dim=-1)

        # Restore sequence dimension
        if is_seq:
            out = out.reshape(B, T, -1)

        return out

    # ── Properties ───────────────────────────────────────────────────────────
    @property
    def num_pairs(self) -> int:
        return len(self.field_pairs)

    @property
    def output_dim(self) -> int:
        """Total output width: num_pairs × interaction_dim."""
        return self.num_pairs * self.interaction_dim

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
