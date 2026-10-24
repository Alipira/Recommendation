"""
Customer VAE.

Squeezes each customer's feature row (profile, payment habits, order history
summary, ...) into a small vector. The mean of the latent distribution (mu)
is used as the customer's node feature in the graph model.

Why a VAE and not a plain autoencoder: the KL term keeps the latent space
smooth, so customers with similar habits land close together, and a new
customer with only a profile still gets a sensible vector.

If you already have CustomerVAE vectors, skip this and load them with
hybridrec.data.erp.load_user_vectors().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    latent_dim: int = 16
    hidden_dim: int = 64
    beta: float = 0.5          # weight of the KL term after warm-up
    warmup_epochs: int = 10    # KL weight grows from 0 to beta over these epochs
    epochs: int = 60
    batch_size: int = 512
    lr: float = 1e-3
    dropout: float = 0.1
    seed: int = 0


class CustomerVAE(nn.Module):
    def __init__(self, n_features: int, cfg: VAEConfig):
        super().__init__()
        h, z = cfg.hidden_dim, cfg.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_features, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(h, h), nn.GELU(),
        )
        self.to_mu = nn.Linear(h, z)
        self.to_logvar = nn.Linear(h, z)
        self.decoder = nn.Sequential(nn.Linear(z, h), nn.GELU(), nn.Linear(h, n_features))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h).clamp(-10.0, 10.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z), mu, logvar


def vae_loss(x, recon, mu, logvar, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rec = F.mse_loss(recon, x, reduction="none").sum(dim=1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
    return rec + beta * kl, rec, kl


def train_vae(
    features: np.ndarray,
    cfg: VAEConfig = VAEConfig(),
    device: str = "cpu",
    log=print,
) -> tuple[CustomerVAE, np.ndarray]:
    """
    Train on a (n_users, n_features) float array (already standardised).
    Returns the model and the mu vectors, shape (n_users, latent_dim).
    """
    torch.manual_seed(cfg.seed)
    x_all = torch.as_tensor(features, dtype=torch.float32)
    model = CustomerVAE(x_all.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    n = len(x_all)
    for epoch in range(cfg.epochs):
        model.train()
        beta = cfg.beta * min(1.0, (epoch + 1) / max(cfg.warmup_epochs, 1))
        perm = torch.randperm(n)
        totals = np.zeros(3)
        for start in range(0, n, cfg.batch_size):
            x = x_all[perm[start:start + cfg.batch_size]].to(device)
            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(x, recon, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            totals += np.array([loss.item(), rec.item(), kl.item()]) * len(x)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == cfg.epochs:
            t = totals / n
            log(f"vae epoch {epoch + 1}/{cfg.epochs}  loss={t[0]:.4f}  recon={t[1]:.4f}  kl={t[2]:.4f}")

    return model, encode_users(model, features, device)


@torch.no_grad()
def encode_users(model: CustomerVAE, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    mu, _ = model.encode(torch.as_tensor(features, dtype=torch.float32, device=device))
    return mu.cpu().numpy()
