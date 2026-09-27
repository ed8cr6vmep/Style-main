"""CPU shape check: WAE 512→64×64×8 and VM-UNet noise head."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models.wae import WideAutoEncoder
from models.vm_unet import ConditionalVMUNet


def test_wae():
    m = WideAutoEncoder(latent_ch=8)
    x = torch.randn(2, 3, 512, 512)
    rec, mu, logvar, z0, skips = m(x)
    assert mu.shape == (2, 8, 64, 64), mu.shape
    assert rec.shape == x.shape
    print("WAE", tuple(mu.shape), tuple(rec.shape), "ok")


def test_vmunet():
    m = ConditionalVMUNet(latent_ch=8)
    z = torch.randn(2, 8, 64, 64)
    t = torch.randint(0, 1000, (2,))
    c = torch.rand(2, 3, 64, 64)
    txt = torch.randn(2, 768)
    eps = m(z, t, c, txt)
    assert eps.shape == z.shape, eps.shape
    print("VM-UNet", tuple(eps.shape), "ok")


if __name__ == "__main__":
    test_wae()
    test_vmunet()
