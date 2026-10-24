"""
LightGCN with node features.

The graph: customers and products are nodes; an edge links a customer to every
product they bought (in training). LightGCN has no weights inside its layers:
each layer just averages a node's neighbours, and the final vector is the mean
over all layers. This spreads "who buys what" through the graph, so two shops
that buy similar things end up close even if they never bought the same SKU.

Node features: each customer's starting vector is

    id_embedding(customer) + Linear(vae_vector(customer))

and products can get the same treatment with item features (e.g. text
embeddings). During training the ID part is randomly dropped for some nodes
(id_dropout), so the model also learns to place nodes from features alone.
That is what lets new customers and new products get useful vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


@dataclass
class GraphConfig:
    dim: int = 64
    n_layers: int = 3
    id_dropout: float = 0.1   # share of nodes whose ID embedding is hidden per step
    l2: float = 1e-4          # weight on the size of the starting vectors
    lr: float = 1e-3
    batch_size: int = 4096
    epochs: int = 50
    eval_every: int = 5
    patience: int = 3         # evaluations without improvement before stopping
    eval_k: int = 20
    seed: int = 0


def build_norm_adj(user_idx: np.ndarray, item_idx: np.ndarray, n_users: int, n_items: int) -> torch.Tensor:
    """
    Symmetric normalised adjacency D^-1/2 A D^-1/2 of the user-item graph,
    as a torch sparse COO tensor of size (n_users + n_items)^2.
    Duplicate (user, item) pairs count once.
    """
    pairs = np.unique(np.stack([user_idx, item_idx], axis=1), axis=0)
    u, i = pairs[:, 0], pairs[:, 1] + n_users
    n = n_users + n_items
    rows = np.concatenate([u, i])
    cols = np.concatenate([i, u])
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    deg = np.asarray(adj.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    d_inv_sqrt[deg > 0] = deg[deg > 0] ** -0.5
    norm = sp.diags(d_inv_sqrt) @ adj @ sp.diags(d_inv_sqrt)
    norm = norm.tocoo()
    index = torch.as_tensor(np.vstack([norm.row, norm.col]), dtype=torch.long)
    values = torch.as_tensor(norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(index, values, (n, n), check_invariants=True).coalesce()


class LightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        cfg: GraphConfig,
        user_features: np.ndarray | None = None,
        item_features: np.ndarray | None = None,
    ):
        super().__init__()
        self.n_users, self.n_items, self.cfg = n_users, n_items, cfg
        self.user_id = nn.Embedding(n_users, cfg.dim)
        self.item_id = nn.Embedding(n_items, cfg.dim)
        nn.init.normal_(self.user_id.weight, std=0.1)
        nn.init.normal_(self.item_id.weight, std=0.1)

        self.user_proj = self.item_proj = None
        if user_features is not None:
            self.register_buffer("user_feat", torch.as_tensor(user_features, dtype=torch.float32))
            self.user_proj = nn.Linear(user_features.shape[1], cfg.dim)
        if item_features is not None:
            self.register_buffer("item_feat", torch.as_tensor(item_features, dtype=torch.float32))
            self.item_proj = nn.Linear(item_features.shape[1], cfg.dim)

        self.adj: torch.Tensor | None = None

    def set_graph(self, adj: torch.Tensor) -> None:
        self.adj = adj.to(self.user_id.weight.device)

    def _start(self, emb: nn.Embedding, proj: nn.Linear | None, feat_name: str) -> torch.Tensor:
        x = emb.weight
        if proj is None:
            return x
        if self.training and self.cfg.id_dropout > 0:
            keep = (torch.rand(x.shape[0], 1, device=x.device) >= self.cfg.id_dropout).float()
            x = x * keep
        return x + proj(getattr(self, feat_name))

    def start_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._start(self.user_id, self.user_proj, "user_feat"),
            self._start(self.item_id, self.item_proj, "item_feat"),
        )

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (final_users, final_items, start_users, start_items)."""
        if self.adj is None:
            raise RuntimeError("call set_graph() first")
        u0, i0 = self.start_vectors()
        x = torch.cat([u0, i0], dim=0)
        total = x
        for _ in range(self.cfg.n_layers):
            x = torch.sparse.mm(self.adj, x)
            total = total + x
        out = total / (self.cfg.n_layers + 1)
        return out[: self.n_users], out[self.n_users:], u0, i0

    def bpr_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bayesian personalised ranking: bought items should score above random ones."""
        fu, fi, u0, i0 = self.propagate()
        eu, ep, en = fu[users], fi[pos], fi[neg]
        margin = (eu * ep).sum(-1) - (eu * en).sum(-1)
        bpr = nn.functional.softplus(-margin).mean()
        reg = (u0[users].pow(2).sum() + i0[pos].pow(2).sum() + i0[neg].pow(2).sum()) / (2 * len(users))
        return bpr + self.cfg.l2 * reg, bpr

    @torch.no_grad()
    def final_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        was = self.training
        self.eval()
        fu, fi, _, _ = self.propagate()
        self.train(was)
        return fu, fi

    @torch.no_grad()
    def vectors_from_features(self, user_features: np.ndarray) -> torch.Tensor:
        """
        Vectors for customers not in the graph (new customers), from their
        feature rows alone. Only works when the model was built with
        user_features. One propagation step through item neighbours is not
        possible without edges, so this is the layer-0 vector.
        """
        if self.user_proj is None:
            raise RuntimeError("model was built without user features")
        x = torch.as_tensor(user_features, dtype=torch.float32, device=self.user_id.weight.device)
        return self.user_proj(x)
