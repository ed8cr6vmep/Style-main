"""Wide AutoEncoder (WAE), manuscript §3.2 / Table 2 / Eq. (6).

Encoder: 512×512×3 → z0 ∈ R^{B×C×64×64}, C=8.
Residual unit: cardinality-32 aggregated residual (ResNeXt-style).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CardinalityResidual(nn.Module):
    """y = x + Σ_{i=1}^{C} F_i(x, W_i), C=32.  Table 2: 256→64→64→256.

    Implemented as a ResNeXt grouped bottleneck (shared 1×1, cardinality-C 3×3).
    """

    def __init__(self, channels: int = 256, bottleneck: int = 64, cardinality: int = 32):
        super().__init__()
        groups = cardinality if bottleneck % cardinality == 0 else 1
        gn_b = 8 if bottleneck % 8 == 0 else 1
        gn_c = 32 if channels % 32 == 0 else 1
        self.f = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1, bias=False),
            nn.GroupNorm(gn_b, bottleneck),
            nn.SiLU(inplace=True),
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=groups, bias=False),
            nn.GroupNorm(gn_b, bottleneck),
            nn.SiLU(inplace=True),
            nn.Conv2d(bottleneck, channels, 1, bias=False),
            nn.GroupNorm(gn_c, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.f(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cardinality: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        inner = max(out_ch, 256)
        self.expand = nn.Conv2d(out_ch, inner, 1) if out_ch != inner else nn.Identity()
        self.res = CardinalityResidual(inner, bottleneck=64, cardinality=cardinality)
        self.compress = nn.Conv2d(inner, out_ch, 1) if out_ch != inner else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.compress(self.res(self.expand(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cardinality: int = 32):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        inner = max(out_ch, 256)
        self.expand = nn.Conv2d(out_ch, inner, 1) if out_ch != inner else nn.Identity()
        self.res = CardinalityResidual(inner, bottleneck=64, cardinality=cardinality)
        self.compress = nn.Conv2d(inner, out_ch, 1) if out_ch != inner else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.compress(self.res(self.expand(x)))


class WideAutoEncoder(nn.Module):
    def __init__(self, latent_ch: int = 8, base_ch: int = 64, cardinality: int = 32):
        super().__init__()
        self.latent_ch = latent_ch
        # 512→256→128→64 so that z0 ∈ R^{B×C×64×64}
        chs = [base_ch, base_ch * 2, base_ch * 4]
        self.stem = nn.Conv2d(3, chs[0], 3, padding=1)
        self.down = nn.ModuleList(
            [
                DownBlock(chs[0], chs[1], cardinality),
                DownBlock(chs[1], chs[2], cardinality),
                DownBlock(chs[2], chs[2], cardinality),
            ]
        )
        self.mid = CardinalityResidual(max(chs[2], 256), 64, cardinality)
        self.to_stat = nn.Conv2d(chs[2], latent_ch * 2, 1)
        self.from_z = nn.Conv2d(latent_ch, chs[2], 1)
        self.up = nn.ModuleList(
            [
                UpBlock(chs[2], chs[2], cardinality),
                UpBlock(chs[2], chs[1], cardinality),
                UpBlock(chs[1], chs[0], cardinality),
            ]
        )
        self.out = nn.Conv2d(chs[0], 3, 3, padding=1)

    def encode(self, x: torch.Tensor):
        h = F.silu(self.stem(x))
        skips = []
        for blk in self.down:
            h = blk(h)
            skips.append(h)
        h = self.mid(h)
        mu, logvar = self.to_stat(h).chunk(2, dim=1)
        return mu, logvar, skips

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, skips=None) -> torch.Tensor:
        h = F.silu(self.from_z(z))
        for i, blk in enumerate(self.up):
            if skips is not None:
                e = skips[-(i + 1)]
                if e.shape[2:] == h.shape[2:] and e.shape[1] == h.shape[1]:
                    h = h + e
            h = blk(h)
        return torch.tanh(self.out(h))

    def forward(self, x: torch.Tensor):
        mu, logvar, skips = self.encode(x)
        z0 = self.reparameterize(mu, logvar)
        rec = self.decode(z0, skips)
        return rec, mu, logvar, z0, skips
