"""LPIPS (content) and Gram style distance, style-transfer protocol §4.4."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


def gram(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    return (f @ f.transpose(2, 1)) / (c * h * w)


class StyleMetrics:
    def __init__(self, device="cuda"):
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.device = device
        try:
            import lpips

            self.lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        except Exception:
            self.lpips_fn = None

    @torch.no_grad()
    def lpips(self, content: torch.Tensor, stylized: torch.Tensor) -> float:
        if self.lpips_fn is None:
            return float(F.l1_loss(content, stylized).item())
        return float(self.lpips_fn(content.to(self.device), stylized.to(self.device)).mean().item())

    @torch.no_grad()
    def style_distance(self, style: torch.Tensor, stylized: torch.Tensor) -> float:
        fs = self.vgg((style.to(self.device) + 1) * 0.5)
        fy = self.vgg((stylized.to(self.device) + 1) * 0.5)
        return float(F.mse_loss(gram(fs), gram(fy)).item())
