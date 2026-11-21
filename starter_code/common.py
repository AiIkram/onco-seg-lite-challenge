"""
Shared helpers for the starter code: data loading, a tiny U-Net, post-processing.
Pure PyTorch - no extra segmentation libraries needed.
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from competition.metrics import IMAGE_SIZE, dice_score  # noqa: E402,F401

DATA = os.path.join(ROOT, "data", "public")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
def load_image(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def load_train():
    meta = pd.read_csv(os.path.join(DATA, "train_metadata.csv"), dtype=str)
    imgs = np.stack([load_image(os.path.join(DATA, "train/images", f"{i}.png")) for i in meta.image_id])
    masks = np.stack([(load_image(os.path.join(DATA, "train/masks", f"{i}.png")) > 0.5).astype(np.float32)
                      for i in meta.image_id])
    return meta, imgs, masks


def load_test():
    ids = pd.read_csv(os.path.join(DATA, "test_ids.csv"), dtype=str).image_id.tolist()
    imgs = np.stack([load_image(os.path.join(DATA, "test/images", f"{i}.png")) for i in ids])
    return ids, imgs


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks=None, size=IMAGE_SIZE, augment=False):
        self.imgs, self.masks, self.size, self.augment = imgs, masks, size, augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = torch.from_numpy(self.imgs[i])[None]
        y = torch.from_numpy(self.masks[i])[None] if self.masks is not None else torch.zeros_like(x)
        if self.augment:
            if random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            # mild intensity jitter (ultrasound gain varies between machines)
            x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05)).clamp(0, 1)
        if self.size != IMAGE_SIZE:
            x = F.interpolate(x[None], size=(self.size, self.size), mode="bilinear", align_corners=False)[0]
            y = (F.interpolate(y[None], size=(self.size, self.size), mode="nearest")[0])
        return (x - 0.5) / 0.25, y


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class TinyUNet(nn.Module):
    """Classic U-Net, small width. base=16, depth=4 -> ~1.9M params (limit is 5M)."""

    def __init__(self, in_ch=1, base=16, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.downs.append(ConvBlock(c, ch))
            c = ch
        self.bottleneck = ConvBlock(chs[-2], chs[-1])
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.dec.append(ConvBlock(ch * 2, ch))
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x):
        skips = []
        for block in self.downs:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Loss / inference                                                             #
# --------------------------------------------------------------------------- #
def bce_dice_loss(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return bce + dice.mean()


@torch.no_grad()
def predict_probs(model, imgs, device, size=IMAGE_SIZE, batch_size=16, tta=True):
    """Returns (N, 256, 256) probabilities at the official resolution."""
    model.eval()
    loader = torch.utils.data.DataLoader(SegDataset(imgs, size=size), batch_size=batch_size)
    out = []
    for x, _ in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x))
        if tta:
            p = (p + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
        if size != IMAGE_SIZE:
            p = F.interpolate(p, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        out.append(p[:, 0].cpu().numpy())
    return np.concatenate(out)


def postprocess(prob, threshold=0.5, min_area=0):
    """Binarise; drop the whole mask if it is smaller than `min_area` pixels
    (cheap way to avoid false-positive blobs on normal scans)."""
    m = (prob > threshold).astype(np.uint8)
    if m.sum() < min_area:
        m[:] = 0
    return m
