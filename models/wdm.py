"""Wide Diffusion Model: WAE + latent DDPM + Conditional VM-UNet.  §3.4."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedules import extract, linear_beta_schedule
from .vm_unet import ConditionalVMUNet
from .wae import WideAutoEncoder


class WideDiffusionModel(nn.Module):
    def __init__(self, cfg: dict, device="cuda"):
        super().__init__()
        wae_cfg = cfg["wae"]
        vm_cfg = cfg["vm_unet"]
        diff = cfg["diffusion"]
        self.device = device
        self.wae = WideAutoEncoder(
            latent_ch=wae_cfg["latent_ch"],
            base_ch=wae_cfg["base_ch"],
            cardinality=wae_cfg["cardinality"],
        )
        self.denoise = ConditionalVMUNet(
            latent_ch=wae_cfg["latent_ch"],
            embed_dim=vm_cfg["embed_dim"],
            depths=tuple(vm_cfg["depths"]),
            d_state=vm_cfg["d_state"],
            cond_dim=vm_cfg["clip_dim"],
            s_color=vm_cfg["s_color"],
            s_text=vm_cfg["s_text"],
        )
        self.T = diff["T"]
        sch = linear_beta_schedule(self.T, diff["beta_start"], diff["beta_end"], "cpu")
        for k, v in sch.items():
            self.register_buffer(k, v)
        self.register_buffer("sqrt_ac", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_om", torch.sqrt(1.0 - self.alphas_cumprod))

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_ac, t, z0.shape) * z0 + extract(self.sqrt_om, t, z0.shape) * noise

    def encode(self, x: torch.Tensor):
        mu, logvar, skips = self.wae.encode(x)
        return mu, skips

    def forward_denoise(self, x, t, color_hsv, text_emb):
        mu, _, skips = self.wae.encode(x)
        z0 = mu
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        eps_hat = self.denoise(z_t, t, color_hsv, text_emb, extra_skips=skips)
        return eps_hat, noise, z0, skips

    @torch.no_grad()
    def dpm_sample(self, x_content, color_hsv, text_emb, steps: int = 20):
        """Algorithm 2: 20-step DPM-Solver-style ancestral loop in latent space."""
        mu, skips = self.encode(x_content)
        z = torch.randn_like(mu)
        times = torch.linspace(self.T - 1, 0, steps, device=z.device).long()
        for i, t_val in enumerate(times):
            t = torch.full((z.size(0),), int(t_val), device=z.device, dtype=torch.long)
            eps = self.denoise(z, t, color_hsv, text_emb, extra_skips=skips)
            ac = extract(self.alphas_cumprod, t, z.shape)
            z0 = (z - torch.sqrt(1 - ac) * eps) / torch.sqrt(ac).clamp_min(1e-6)
            if i == steps - 1:
                z = z0
                break
            t_next = times[i + 1]
            ac_n = self.alphas_cumprod[int(t_next)].view(1, 1, 1, 1)
            z = torch.sqrt(ac_n) * z0 + torch.sqrt(1 - ac_n) * eps
        return self.wae.decode(z, skips)
