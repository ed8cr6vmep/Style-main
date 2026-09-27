"""DDPM linear schedule, manuscript β_t = linear(1e-4, 2e-2, T), T=1000."""
from __future__ import annotations

import torch


def linear_beta_schedule(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alphas_cumprod": alphas_cumprod}


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    b = t.shape[0]
    out = a.gather(0, t)
    return out.view(b, *((1,) * (len(x_shape) - 1)))
