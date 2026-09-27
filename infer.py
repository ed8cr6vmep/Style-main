"""Algorithm 2 — WDM inference (manuscript §3.4).

encode content x → (z-skips {E_i});
sample z_T ~ N(0, I);
denoise 20 DPM-Solver steps with (c, text);
decode D(z_0; E_i) → 512×512.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

from models import WideDiffusionModel
from utils import CLIPText, load_cfg, load_ckpt, set_seed


def load_rgb(path, size=512):
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)


def rgb_to_hsv64(x):
    import torch.nn.functional as F

    x01 = (x + 1) * 0.5
    mx, mn = x01.max(1).values, x01.min(1).values
    s = torch.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), torch.zeros_like(mx))
    hsv = torch.stack([torch.zeros_like(s), s, mx], 1)
    return F.interpolate(hsv, size=(64, 64), mode="bilinear", align_corners=False)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wdm.yaml")
    ap.add_argument("--ckpt", default="checkpoints/wdm.pt")
    ap.add_argument("--content", required=True)
    ap.add_argument("--prompt", default="a traditional Chinese ornamental pattern")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="outputs/stylized.png")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WideDiffusionModel(cfg, device).to(device)
    if Path(args.ckpt).is_file():
        load_ckpt(args.ckpt, model, map_location=device)
    model.eval()
    x = load_rgb(args.content, cfg["data"]["image_size"]).to(device)
    color = rgb_to_hsv64(x)
    text = CLIPText(device)([args.prompt])
    steps = args.steps or cfg["diffusion"]["infer_steps"]
    y = model.dpm_sample(x, color, text, steps=steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(y * 0.5 + 0.5, args.out)


if __name__ == "__main__":
    main()
