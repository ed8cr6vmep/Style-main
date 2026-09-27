"""CPU wrap check for WideDiffusionModel + Eq. (9) loss."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from losses import total_loss
from models.wdm import WideDiffusionModel
from utils import load_cfg


def test_wdm():
    cfg = load_cfg(str(Path(__file__).resolve().parents[1] / "configs" / "wdm.yaml"))
    m = WideDiffusionModel(cfg, "cpu")
    x = torch.randn(1, 3, 512, 512)
    color = torch.rand(1, 3, 64, 64)
    text = torch.randn(1, 768)
    t = torch.randint(0, 1000, (1,))
    eps_hat, eps, z0, skips = m.forward_denoise(x, t, color, text)
    rec = m.wae.decode(z0, skips)
    loss, logs = total_loss(eps_hat, eps, x, rec)
    assert eps_hat.shape == z0.shape
    assert rec.shape == x.shape
    assert loss.ndim == 0
    print("WDM", tuple(z0.shape), logs, "ok")


if __name__ == "__main__":
    test_wdm()
