"""Conditional VM-UNet denoiser, manuscript §3.3 / Fig. 4 / Eq. (7)(8).

Input : z_t ⊕ τ(t) ∈ R^{B×C×64×64}
Output: ε̂ of the same shape (noise residual, not a class map).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vss import VSSBlock, PatchEmbed, PatchMerging, PatchExpanding


class TimestepEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class CrossAttention(nn.Module):
    """SA(Q,K,V)=softmax(QK^T/√d)V,  Q←F,  (K,V)←[c∥t]   Eq. (7)."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.scale = dim ** -0.5
        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Linear(cond_dim, dim * 2)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        q = self.q(feat).flatten(2).transpose(1, 2)
        k, v = self.kv(cond).chunk(2, dim=-1)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return self.proj(out)


class AdaGN(nn.Module):
    """AdaGN(F)=γ(c)⊙(F−μ)/σ + β(c)   Eq. (8)."""

    def __init__(self, dim: int, cond_ch: int):
        super().__init__()
        self.gn = nn.GroupNorm(8, dim, affine=False)
        self.to_gb = nn.Conv2d(cond_ch, dim * 2, 1)

    def forward(self, feat: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
        if color.shape[-2:] != feat.shape[-2:]:
            color = F.interpolate(color, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.to_gb(color).chunk(2, dim=1)
        return gamma * self.gn(feat) + beta


class DecoderStage(nn.Module):
    def __init__(self, dim: int, cond_dim: int, color_ch: int, n_vss: int = 2, d_state: int = 16, time_dim: int = 96):
        super().__init__()
        self.blocks = nn.ModuleList([VSSBlock(dim, d_state) for _ in range(n_vss)])
        self.attn = CrossAttention(dim, cond_dim)
        self.adagn = AdaGN(dim, color_ch)
        self.film = nn.Linear(time_dim, dim * 2)

    def forward(self, x, cond, color, t_emb, s_color=1.0, s_text=0.7):
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        for blk in self.blocks:
            x = blk(x)
        x = x + s_text * self.attn(x, cond)
        x = x + s_color * self.adagn(x, color)
        return x


class ConditionalVMUNet(nn.Module):
    def __init__(
        self,
        latent_ch: int = 8,
        embed_dim: int = 96,
        depths=(2, 2, 2, 2),
        d_state: int = 16,
        cond_dim: int = 768,
        s_color: float = 1.0,
        s_text: float = 0.7,
    ):
        super().__init__()
        self.s_color = s_color
        self.s_text = s_text
        self.time = TimestepEmbed(embed_dim)
        self.color_in = nn.Conv2d(3, latent_ch, 1)
        self.text_mlp = nn.Sequential(nn.Linear(cond_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.patch = PatchEmbed(latent_ch, embed_dim)
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.enc_vss = nn.ModuleList()
        self.merge = nn.ModuleList()
        for i, n in enumerate(depths):
            self.enc_vss.append(nn.Sequential(*[VSSBlock(dims[i], d_state) for _ in range(n)]))
            if i < 3:
                self.merge.append(PatchMerging(dims[i]))
        self.skip_proj = nn.ModuleList([nn.Conv2d(d, d, 1) for d in dims[:3]])
        self.expand = nn.ModuleList(
            [PatchExpanding(dims[3]), PatchExpanding(dims[2]), PatchExpanding(dims[1])]
        )
        self.dec = nn.ModuleList(
            [
                DecoderStage(dims[2], embed_dim, latent_ch, depths[2], d_state, embed_dim),
                DecoderStage(dims[1], embed_dim, latent_ch, depths[1], d_state, embed_dim),
                DecoderStage(dims[0], embed_dim, latent_ch, depths[0], d_state, embed_dim),
            ]
        )
        self.noise_head = nn.Conv2d(embed_dim, latent_ch, 1)

    def forward(self, z_t, t, color_hsv, text_emb, extra_skips=None):
        """
        z_t:      B,C,64,64
        t:        B
        color_hsv:B,3,64,64
        text_emb: B,768
        extra_skips: optional WAE multi-scale {E_i}, 1×1 projected by caller or here.
        """
        t_emb = self.time(t)
        color = self.color_in(color_hsv)
        cond = self.text_mlp(text_emb)[:, None, :]
        x = self.patch(z_t)
        encs = []
        for i, blk in enumerate(self.enc_vss):
            x = blk(x)
            if i < 3:
                encs.append(x)
                x = self.merge[i](x)
        for i, (exp, dec) in enumerate(zip(self.expand, self.dec)):
            x = exp(x)
            skip = self.skip_proj[2 - i](encs[2 - i])
            if extra_skips is not None and i < len(extra_skips):
                e = extra_skips[-(i + 1)]
                if e.shape[2:] == x.shape[2:]:
                    if e.shape[1] != x.shape[1]:
                        # lazy 1×1 to align WAE skip channels
                        e = F.interpolate(e.mean(1, keepdim=True).expand(-1, x.shape[1], -1, -1), size=x.shape[2:])
                    skip = skip + e
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")
            x = dec(x + skip, cond, color, t_emb, self.s_color, self.s_text)
        return self.noise_head(x)
