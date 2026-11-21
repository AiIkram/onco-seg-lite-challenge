#!/usr/bin/env python3
"""
ORGANISER - generate a FAKE BUSI-shaped dataset to dry-run the whole pipeline
(prepare_data -> encrypt -> CI scoring -> leaderboard) before touching real data.

    python scripts/make_dummy_busi.py --out /tmp/Dummy_BUSI --n 300

Images: speckled "ultrasound-like" texture with dark elliptical lesions
(benign = smooth, malignant = irregular), normal = no lesion. A few exact
duplicates are planted to exercise the de-duplication logic.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageFilter


def speckle(rng, h, w):
    base = rng.gamma(2.0, 0.5, size=(h, w))
    img = Image.fromarray(np.clip(base * 70, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    arr = np.asarray(img, dtype=np.float32)
    arr *= np.linspace(1.1, 0.6, h)[:, None]  # depth attenuation
    return arr


def lesion_mask(rng, h, w, irregular):
    yy, xx = np.mgrid[:h, :w]
    cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.25, 0.75) * w
    ry, rx = rng.uniform(0.06, 0.18) * h, rng.uniform(0.08, 0.22) * w
    ang = np.arctan2(yy - cy, xx - cx)
    r = 1.0
    if irregular:
        for k in (3, 5, 7):
            r = r + rng.uniform(0.05, 0.18) * np.sin(k * ang + rng.uniform(0, 6.28))
    return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= r ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=780)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    counts = {"benign": round(args.n * 0.56), "malignant": round(args.n * 0.27)}
    counts["normal"] = args.n - counts["benign"] - counts["malignant"]
    for cls, n in counts.items():
        d = os.path.join(args.out, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(1, n + 1):
            h, w = int(rng.integers(420, 620)), int(rng.integers(450, 650))
            img = speckle(rng, h, w)
            mask = np.zeros((h, w), bool)
            if cls != "normal":
                mask = lesion_mask(rng, h, w, irregular=(cls == "malignant"))
                img[mask] *= rng.uniform(0.15, 0.4)
                if rng.random() < 0.1:  # second lesion -> separate _mask_1 file
                    m2 = lesion_mask(rng, h, w, irregular=False) & ~mask
                    img[m2] *= 0.3
                    Image.fromarray(m2.astype(np.uint8) * 255).save(os.path.join(d, f"{cls} ({i})_mask_1.png"))
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(d, f"{cls} ({i}).png"))
            Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(d, f"{cls} ({i})_mask.png"))
        # plant 2 duplicates per class
        for j in range(2):
            src = os.path.join(d, f"{cls} ({j + 1}).png")
            k = n + j + 1
            Image.open(src).save(os.path.join(d, f"{cls} ({k}).png"))
            Image.open(os.path.join(d, f"{cls} ({j + 1})_mask.png")).save(os.path.join(d, f"{cls} ({k})_mask.png"))
    print(f"Dummy BUSI written to {args.out}: {counts} (+2 duplicates per class)")


if __name__ == "__main__":
    main()
