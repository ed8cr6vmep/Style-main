"""Algorithm 1 — WDM training (manuscript §3.4 / Eq. 9–10).

1. Pre-train WAE with L_recon = ||x − D(E(x))||_2^2.
2. Jointly minimise L = L_DDPM + 0.1 L_recon + 0.05 L_color
   on CTDPD with AdamW (lr=5e-5, β=(0.9, 0.99)), batch 8, T=1000,
   linear β_t ∈ [1e-4, 2e-2], base seed 42.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from losses import total_loss
from models import WideDiffusionModel
from utils import CLIPText, load_cfg, save_ckpt, set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wdm.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def make_opt(params, cfg):
    return torch.optim.AdamW(
        params,
        lr=cfg["train"]["lr"],
        betas=tuple(cfg["train"]["betas"]),
        weight_decay=cfg["train"]["weight_decay"],
    )


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["train"]["seed"])

    from data.ctdpd import make_loader

    root = args.data or cfg["data"]["root"]
    loader = make_loader(
        root,
        "train",
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
        cfg["data"]["image_size"],
        cfg["train"]["seed"],
    )
    model = WideDiffusionModel(cfg, device).to(device)
    clip = CLIPText(device)
    l1, l2 = cfg["train"]["lambda_recon"], cfg["train"]["lambda_color"]
    sat = cfg["train"]["color_sat_thresh"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- stage 1: WAE pre-train ----
    opt_w = make_opt(model.wae.parameters(), cfg)
    for epoch in range(cfg["train"]["wae_pretrain_epochs"]):
        model.wae.train()
        pbar = tqdm(loader, desc=f"WAE {epoch+1}")
        for batch in pbar:
            x = batch["image"].to(device)
            rec, *_ = model.wae(x)
            loss = torch.nn.functional.mse_loss(rec, x)
            opt_w.zero_grad()
            loss.backward()
            opt_w.step()
            pbar.set_postfix(L_recon=f"{loss.item():.4f}")
        save_ckpt(out / "wae_last.pt", model.wae, epoch)

    # ---- stage 2: joint Eq. (9) ----
    opt = make_opt(model.parameters(), cfg)
    for epoch in range(cfg["train"]["joint_epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"WDM {epoch+1}")
        for batch in pbar:
            x = batch["image"].to(device)
            color = batch["color"].to(device)
            text = clip(batch["prompt"])
            t = torch.randint(0, model.T, (x.size(0),), device=device)
            eps_hat, eps, z0, skips = model.forward_denoise(x, t, color, text)
            x_rec = model.wae.decode(z0, skips)
            loss, logs = total_loss(eps_hat, eps, x, x_rec, None, l1, l2, sat)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(**{k: f"{v:.3f}" for k, v in logs.items()})
        save_ckpt(out / "wdm_last.pt", model, epoch)
    save_ckpt(out / "wdm.pt", model, cfg["train"]["joint_epochs"])


if __name__ == "__main__":
    main()
