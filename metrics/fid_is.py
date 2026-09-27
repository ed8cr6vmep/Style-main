"""FID (Eq. 11) and IS (Eq. 12), ImageNet Inception-v3, manuscript §4.1."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3


def _inception(device):
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    m.fc = torch.nn.Identity()
    m.eval().to(device)
    return m


def _inception_logits(device):
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    m.eval().to(device)
    return m


@torch.no_grad()
def _feats(model, images, device):
    # images: [-1,1] BCHW
    x = F.interpolate((images + 1) * 0.5, size=(299, 299), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5
    return model(x.to(device)).detach().cpu().numpy()


def fid_score(real: torch.Tensor, fake: torch.Tensor, device="cuda") -> float:
    """Eq. (11): ||μ_r−μ_g||² + Tr(Σ_r+Σ_g−2(Σ_r Σ_g)^{1/2})."""
    model = _inception(device)
    fr, fg = _feats(model, real, device), _feats(model, fake, device)
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    sig_r = np.cov(fr, rowvar=False)
    sig_g = np.cov(fg, rowvar=False)
    diff = mu_r - mu_g
    covmean = _sqrtm(sig_r @ sig_g)
    return float(diff @ diff + np.trace(sig_r + sig_g - 2 * covmean))


def inception_score(fake: torch.Tensor, device="cuda", splits: int = 10) -> float:
    """Eq. (12): exp(E_X[KL(p(y|X)||p(y))])."""
    model = _inception_logits(device)
    x = F.interpolate((fake + 1) * 0.5, size=(299, 299), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5
    preds = []
    for i in range(0, x.size(0), 8):
        logits = model(x[i : i + 8].to(device))
        preds.append(F.softmax(logits, dim=1).cpu().numpy())
    p = np.concatenate(preds, 0)
    scores = []
    n = p.shape[0]
    for k in range(splits):
        sl = p[k * n // splits : (k + 1) * n // splits]
        py = sl.mean(0, keepdims=True)
        kl = sl * (np.log(sl + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(kl.sum(1).mean()))
    return float(np.mean(scores))


def _sqrtm(mat):
    w, v = np.linalg.eigh((mat + mat.T) / 2)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.T
