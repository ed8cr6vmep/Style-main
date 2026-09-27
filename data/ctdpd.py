"""CTDPD loader. 6 categories, 8:1:1 stratified split, 512², manuscript §4.1."""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CATEGORIES = [
    "mountains_rivers",
    "birds_flowers",
    "auspicious_animals",
    "geometrics",
    "calligraphy",
    "paper_cuttings",
]


class CTDPD(Dataset):
    def __init__(self, root: str, split: str = "train", image_size: int = 512, seed: int = 42):
        self.root = Path(root)
        self.split = split
        self.tf = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.aug = split == "train"
        files = []
        for i, cat in enumerate(CATEGORIES):
            folder = self.root / cat
            imgs = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpeg"))
            rng = random.Random(seed + i)
            rng.shuffle(imgs)
            n = len(imgs)
            n_tr, n_va = int(0.8 * n), int(0.1 * n)
            if split == "train":
                chosen = imgs[:n_tr]
            elif split == "val":
                chosen = imgs[n_tr : n_tr + n_va]
            else:
                chosen = imgs[n_tr + n_va :]
            files.extend([(p, i, cat) for p in chosen])
        self.items = files

    def __len__(self):
        return max(len(self.items), 1)

    def __getitem__(self, idx):
        if not self.items:
            img = Image.new("RGB", (512, 512), (128, 128, 128))
            lab, cat = 0, CATEGORIES[0]
        else:
            path, lab, cat = self.items[idx % len(self.items)]
            img = Image.open(path).convert("RGB")
        if self.aug:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.rotate(random.uniform(-45, 45), resample=Image.BILINEAR, fillcolor=(128, 128, 128))
        x = self.tf(img)
        hsv = _rgb_tensor_to_hsv64(x)
        prompt = f"a traditional Chinese ornamental pattern of {cat.replace('_', ' ')}"
        return {"image": x, "color": hsv, "label": lab, "prompt": prompt, "category": cat}


def _rgb_tensor_to_hsv64(x):
    import torch
    import torch.nn.functional as F

    x01 = (x + 1) * 0.5
    r, g, b = x01[0], x01[1], x01[2]
    mx, mn = x01.max(0).values, x01.min(0).values
    s = torch.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), torch.zeros_like(mx))
    v = mx
    h = torch.zeros_like(mx)
    hsv = torch.stack([h, s, v], 0)
    return F.interpolate(hsv.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False).squeeze(0)


def make_loader(root, split, batch_size, num_workers=4, image_size=512, seed=42):
    ds = CTDPD(root, split, image_size, seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=split == "train", num_workers=num_workers, drop_last=True)
