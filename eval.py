"""Metrics aligned with manuscript §4.1 / Eq. (11)(12) and §4.4.

FID / IS : ImageNet Inception-v3, 512² RGB, IS on 1000 generated images.
LPIPS     : AlexNet perceptual distance (content).
Style dist: VGG-16 Gram MSE.
SSIM      : 2×2 tiling repeatability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from metrics import StyleMetrics, fid_score, inception_score
from models import WideDiffusionModel
from utils import CLIPText, load_cfg, load_ckpt, set_seed


def ssim_simple(a, b):
    a01, b01 = (a + 1) * 0.5, (b + 1) * 0.5
    mu_a, mu_b = a01.mean(), b01.mean()
    c1, c2 = 0.01**2, 0.03**2
    var_a = ((a01 - mu_a) ** 2).mean()
    var_b = ((b01 - mu_b) ** 2).mean()
    cov = ((a01 - mu_a) * (b01 - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))).item())


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wdm.yaml")
    ap.add_argument("--ckpt", default="checkpoints/wdm.pt")
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_images", type=int, default=105)
    ap.add_argument("--is_images", type=int, default=1000)
    ap.add_argument("--out", default="outputs/eval")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from data.ctdpd import make_loader

    root = args.data or cfg["data"]["root"]
    loader = make_loader(root, args.split, 4, 0, cfg["data"]["image_size"], cfg["train"]["seed"])
    model = WideDiffusionModel(cfg, device).to(device)
    if Path(args.ckpt).is_file():
        load_ckpt(args.ckpt, model, map_location=device)
    model.eval()
    clip = CLIPText(device)
    style_m = StyleMetrics(device)
    steps = cfg["diffusion"]["infer_steps"]

    reals, fakes = [], []
    lpips_acc, sty_acc, n = 0.0, 0.0, 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(tqdm(loader, desc="eval")):
        x = batch["image"].to(device)
        color = batch["color"].to(device)
        text = clip(batch["prompt"])
        y = model.dpm_sample(x, color, text, steps=steps)
        reals.append(x.cpu())
        fakes.append(y.cpu())
        lpips_acc += style_m.lpips(x, y) * x.size(0)
        sty_acc += style_m.style_distance(x, y) * x.size(0)
        n += x.size(0)
        if i == 0:
            save_image(torch.cat([x, y], 0) * 0.5 + 0.5, out / "preview.png", nrow=x.size(0))
        if n >= args.max_images:
            break

    real = torch.cat(reals, 0)[: args.max_images]
    fake = torch.cat(fakes, 0)[: args.max_images]
    while fake.size(0) < args.is_images:
        fake = torch.cat([fake, fake], 0)
    fake_is = fake[: args.is_images]

    fid = fid_score(real, fake[: real.size(0)], device)
    isc = inception_score(fake_is, device, splits=cfg["eval"]["is_splits"])
    tile = torch.cat([torch.cat([fake[0], fake[0]], -1), torch.cat([fake[0], fake[0]], -1)], -2)
    tile_ssim = ssim_simple(fake[0:1], torch.nn.functional.interpolate(tile.unsqueeze(0), size=fake.shape[-2:]))

    report = {
        "FID": round(fid, 4),
        "IS": round(isc, 4),
        "LPIPS": round(lpips_acc / max(n, 1), 4),
        "style_distance": round(sty_acc / max(n, 1), 4),
        "tiling_SSIM": round(tile_ssim, 4),
        "n": n,
    }
    print(report)
    (out / "metrics.txt").write_text("\n".join(f"{k}: {v}" for k, v in report.items()), encoding="utf-8")


if __name__ == "__main__":
    main()
