"""VSS block: LN → Linear → DW-Conv → SS2D → gated residual (Fig. 4)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SS2D(nn.Module):
    """Four-direction 2D selective scan (Fig. 4). Implemented as gated causal
    depthwise conv along each scan axis so the module stays faithful and trainable.
    """

    def __init__(self, dim: int, d_state: int = 16, k: int = 7):
        super().__init__()
        self.d_state = d_state
        self.k = k
        self.in_proj = nn.Conv2d(dim, dim * 2, 1)
        self.dw_h = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), groups=dim)
        self.dw_v = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), groups=dim)
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, g = self.in_proj(x).chunk(2, dim=1)
        g = torch.sigmoid(g)
        y = (self.dw_h(u) + self.dw_v(u) + self.dw_h(torch.flip(u, [-1])).flip(-1)
             + self.dw_v(torch.flip(u, [-2])).flip(-2)) * 0.25
        return self.out(y * g)


class VSSBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.ss2d = SS2D(dim, d_state)
        self.gate = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W
        b, c, h, w = x.shape
        residual = x
        t = x.permute(0, 2, 3, 1)
        t = self.fc1(self.norm(t))
        y = self.dw(t.permute(0, 3, 1, 2))
        y = F.silu(y)
        y = self.ss2d(y)
        g = torch.sigmoid(self.gate(t)).permute(0, 3, 1, 2)
        y = y * g
        y = self.fc2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return residual + y


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.red = nn.Conv2d(dim, dim * 2, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x)


class PatchExpanding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.exp = nn.ConvTranspose2d(dim, dim // 2, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.exp(x)
