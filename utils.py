"""Shared helpers: config, seed, CLIP ViT-L/14 text embedding (768-d)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_cfg(path: str = "configs/wdm.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CLIPText:
    """CLIP ViT-L/14 → 768-d, manuscript §3.3. Falls back to a hash embed if CLIP is absent."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.tokenize = None
        try:
            import clip

            self.model, _ = clip.load("ViT-L/14", device=device)
            self.model.eval()
            self.tokenize = clip.tokenize
        except Exception:
            self.model = None

    @torch.no_grad()
    def __call__(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        if self.model is not None:
            tok = self.tokenize(list(prompts), truncate=True).to(self.device)
            return self.model.encode_text(tok).float()
        embs = []
        for p in prompts:
            g = torch.Generator(device="cpu")
            g.manual_seed(int.from_bytes(p.encode("utf-8")[:8].ljust(8, b"\0"), "little") % (2**31))
            embs.append(torch.randn(768, generator=g))
        return torch.stack(embs, 0).to(self.device)


def save_ckpt(path, model, epoch, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "extra": extra or {}}, path)


def load_ckpt(path, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)
    return ckpt
