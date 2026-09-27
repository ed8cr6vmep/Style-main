"""Eq. (9)(10): L = L_DDPM + 0.1 L_recon + 0.05 L_color."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rgb_to_hsv_s(x: torch.Tensor) -> torch.Tensor:
    """x in [-1,1] → saturation in [0,1]."""
    x01 = (x + 1) * 0.5
    mx, mn = x01.max(1).values, x01.min(1).values
    return torch.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), torch.zeros_like(mx))


def loss_ddpm(eps_hat: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(eps_hat, eps)


def loss_recon(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def loss_color(x: torch.Tensor, y: torch.Tensor, sat_thresh: float = 0.25) -> torch.Tensor:
    """L_color = ||M ⊙ (y − x)||_2^2,  M = 1[S_HSV > 0.25]."""
    m = (rgb_to_hsv_s(x) > sat_thresh).float().unsqueeze(1)
    return ((m * (y - x)) ** 2).mean()


def total_loss(eps_hat, eps, x, x_rec, y=None, l1=0.1, l2=0.05, sat_thresh=0.25):
    ld = loss_ddpm(eps_hat, eps)
    lr = loss_recon(x, x_rec)
    lc = loss_color(x, y if y is not None else x_rec, sat_thresh)
    return ld + l1 * lr + l2 * lc, {"L_DDPM": ld.item(), "L_recon": lr.item(), "L_color": lc.item()}
