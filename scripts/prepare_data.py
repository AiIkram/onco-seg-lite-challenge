#!/usr/bin/env python3
"""
ORGANISER ONLY - build the public/private challenge data from the raw BUSI release.

Raw input (as distributed, e.g. "Dataset_BUSI_with_GT/"):
    benign/    benign (1).png, benign (1)_mask.png, benign (1)_mask_1.png, ...
    malignant/ ...
    normal/    ...

    python scripts/prepare_data.py --raw /path/to/Dataset_BUSI_with_GT --secret "<long random string>"

What it does
    1. Merges multi-lesion masks (_mask, _mask_1, ...) into one binary mask.
    2. Groups exact / near-duplicate images (perceptual dHash) so that duplicates
       never straddle train and test - BUSI is known to contain duplicates.
    3. Stratified 80/20 split by class, at duplicate-group level.
    4. Resizes everything to 256x256 grayscale (bilinear for images, nearest for masks).
    5. Anti-contamination (default `--modification standard`): lesion-preserving random
       re-crop + gamma + speckle on every image, so no file is pixel-identical to BUSI.
       `--test-shift` adds a simulated device shift to the hidden test set (Hard variant).
    6. Renames every case to an opaque ID = HMAC(secret, original filename) so the
       original BUSI file names are not recoverable from the public repo.

Outputs
    data/public/train/images/<id>.png      (commit)
    data/public/train/masks/<id>.png       (commit; 0 = background, 255 = lesion)
    data/public/train_metadata.csv         (commit; image_id, diagnosis)
    data/public/test/images/<id>.png       (commit)
    data/public/test_ids.csv               (commit)
    data/public/sample_submission.csv      (commit; all-empty masks)
    data/private/test_ground_truth.npz     (NEVER commit -> encrypt with scripts/encrypt_ground_truth.py)
    data/private/test_mapping.csv          (NEVER commit; id <-> original file, for your analysis)

Keep --secret private (password manager). The same secret + same raw data
reproduces the exact same split.
"""
import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from competition.metrics import IMAGE_SIZE  # noqa: E402

CLASSES = ("benign", "malignant", "normal")
MASK_RE = re.compile(r"^(?P<base>.+?)_mask(?:_\d+)?\.png$", re.IGNORECASE)


def collect_cases(raw_dir):
    cases = []
    for cls in CLASSES:
        d = os.path.join(raw_dir, cls)
        if not os.path.isdir(d):
            sys.exit(f"Missing class folder: {d}")
        files = sorted(os.listdir(d))
        masks = defaultdict(list)
        images = []
        for f in files:
            if not f.lower().endswith(".png"):
                continue
            m = MASK_RE.match(f)
            if m:
                masks[m.group("base")].append(os.path.join(d, f))
            else:
                images.append(f)
        for f in images:
            base = f[:-4]
            if not masks.get(base):
                print(f"  ! no mask for {cls}/{f} - skipped")
                continue
            cases.append({"orig": f"{cls}/{f}", "cls": cls,
                          "img_path": os.path.join(d, f), "mask_paths": masks[base]})
    return cases


def load_case(case):
    """Native-resolution grayscale image (uint8) and merged binary mask (bool)."""
    img = Image.open(case["img_path"]).convert("L")
    mask = np.zeros((img.height, img.width), dtype=bool)
    for mp in case["mask_paths"]:
        m = Image.open(mp).convert("L")
        if m.size != img.size:
            m = m.resize(img.size, Image.NEAREST)
        mask |= np.asarray(m) > 127
    return np.asarray(img), mask


def resize_pair(img, mask):
    img_r = Image.fromarray(img).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    mask_r = Image.fromarray(mask.astype(np.uint8) * 255).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return np.asarray(img_r), (np.asarray(mask_r) > 127).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Anti-contamination modification (GNN-CB 3.1 analogue: "avoid canonical      #
# configurations", e.g. noise injection, cross-site shift). Every image is     #
# re-framed and re-textured so that it is no longer pixel-identical to the     #
# public BUSI release; masks undergo the identical geometric transform.        #
# --------------------------------------------------------------------------- #
def case_rng(secret, orig):
    seed = int.from_bytes(hmac.new(secret.encode(), ("mod|" + orig).encode(), hashlib.sha256).digest()[:8], "big")
    return np.random.default_rng(seed)


def reframe(img, mask, rng, scale=(0.82, 0.95)):
    """Random crop (keeping the whole lesion inside) - changes framing and aspect ratio."""
    h, w = img.shape
    ch, cw = int(h * rng.uniform(*scale)), int(w * rng.uniform(*scale))
    if mask.any():
        ys, xs = np.nonzero(mask)
        y0_min, y0_max = max(0, ys.max() + 1 - ch), min(h - ch, ys.min())
        x0_min, x0_max = max(0, xs.max() + 1 - cw), min(w - cw, xs.min())
        if y0_min > y0_max:  # lesion taller than crop -> grow crop to fit it
            ch = min(h, ys.max() + 1 - ys.min()); y0_min = y0_max = max(0, min(ys.min(), h - ch))
        if x0_min > x0_max:
            cw = min(w, xs.max() + 1 - xs.min()); x0_min = x0_max = max(0, min(xs.min(), w - cw))
    else:
        y0_min, y0_max, x0_min, x0_max = 0, h - ch, 0, w - cw
    y0 = int(rng.integers(y0_min, y0_max + 1))
    x0 = int(rng.integers(x0_min, x0_max + 1))
    return img[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]


def retexture(img, rng, gamma=(0.85, 1.15), speckle_shape=30.0, blur=0.0):
    """Gamma change + multiplicative speckle (gamma-distributed, mean 1) + optional blur."""
    x = img.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 0, 1), rng.uniform(*gamma))
    x = x * rng.gamma(speckle_shape, 1.0 / speckle_shape, size=x.shape)
    out = Image.fromarray(np.clip(x * 255, 0, 255).astype(np.uint8))
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(out)


def modify(img, mask, rng, mode, test_shift):
    if mode == "none":
        return img, mask
    img, mask = reframe(img, mask, rng)
    img = retexture(img, rng)
    if test_shift:  # simulated acquisition / device shift on the hidden test set only
        img = retexture(img, rng, gamma=(0.7, 1.4), speckle_shape=10.0, blur=rng.uniform(0.6, 1.2))
    return img, mask


def dhash(img_arr, size=16):
    small = np.asarray(Image.fromarray(img_arr).resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (small[:, 1:] > small[:, :-1]).reshape(-1)


def group_duplicates(hashes, max_hamming):
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    H = np.stack(hashes).astype(np.uint8)
    for i in range(n):
        dist = (H[i + 1:] != H[i]).sum(axis=1)
        for j in np.flatnonzero(dist <= max_hamming) + i + 1:
            parent[find(i)] = find(int(j))
    return [find(i) for i in range(n)]


def opaque_id(secret, orig):
    return "img_" + hmac.new(secret.encode(), orig.encode(), hashlib.sha256).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to Dataset_BUSI_with_GT")
    ap.add_argument("--secret", required=True, help="Private string used for IDs and split")
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--modification", choices=["standard", "none"], default="standard",
                    help="standard = lesion-preserving re-crop + gamma + speckle on ALL images (anti-contamination)")
    ap.add_argument("--test-shift", action="store_true",
                    help="HARD variant: extra acquisition shift (gamma, strong speckle, blur) on test images only")
    ap.add_argument("--dup_hamming", type=int, default=6,
                    help="Max dHash Hamming distance (of 256 bits) to treat two images as duplicates")
    args = ap.parse_args()

    if len(args.secret) < 12:
        sys.exit("--secret should be at least 12 characters")

    cases = collect_cases(args.raw)
    print(f"Found {len(cases)} image/mask pairs")
    if not cases:
        sys.exit("Nothing to do.")

    natives, hashes = [], []
    for c in cases:
        im, m = load_case(c)
        natives.append((im, m))
        hashes.append(dhash(resize_pair(im, m)[0]))  # hash the UNMODIFIED image
        c["id"] = opaque_id(args.secret, c["orig"])

    groups = group_duplicates(hashes, args.dup_hamming)
    n_groups = len(set(groups))
    print(f"Duplicate grouping: {len(cases)} images -> {n_groups} groups "
          f"({len(cases) - n_groups} images share a group with another)")

    # Stratified split at group level (group class = majority class of its members)
    seed = int.from_bytes(hashlib.sha256(args.secret.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    group_members = defaultdict(list)
    for idx, g in enumerate(groups):
        group_members[g].append(idx)
    by_cls = defaultdict(list)
    for g, members in group_members.items():
        cls = pd.Series([cases[i]["cls"] for i in members]).mode()[0]
        by_cls[cls].append(g)

    test_idx = set()
    for cls in CLASSES:
        gs = sorted(by_cls[cls])
        rng.shuffle(gs)
        target = round(args.test_frac * sum(len(group_members[g]) for g in gs))
        taken = 0
        for g in gs:
            if taken >= target:
                break
            test_idx.update(group_members[g])
            taken += len(group_members[g])

    # Write outputs
    pub = os.path.join(ROOT, "data", "public")
    priv = os.path.join(ROOT, "data", "private")
    for sub in ("train", "test"):
        shutil.rmtree(os.path.join(pub, sub), ignore_errors=True)
    for d in ("train/images", "train/masks", "test/images"):
        os.makedirs(os.path.join(pub, d), exist_ok=True)
    os.makedirs(priv, exist_ok=True)

    imgs, masks = [], []
    for i, c in enumerate(cases):
        im, m = modify(*natives[i], case_rng(args.secret, c["orig"]), args.modification,
                       test_shift=args.test_shift and i in test_idx)
        im, m = resize_pair(im, m)
        imgs.append(im)
        masks.append(m)
    del natives

    train_rows, test_rows = [], []
    gt_ids, gt_masks, gt_cls = [], [], []
    for i, c in enumerate(cases):
        if i in test_idx:
            Image.fromarray(imgs[i]).save(os.path.join(pub, "test/images", c["id"] + ".png"))
            test_rows.append({"image_id": c["id"], "diagnosis": c["cls"], "original_file": c["orig"]})
            gt_ids.append(c["id"]); gt_masks.append(masks[i]); gt_cls.append(c["cls"])
        else:
            Image.fromarray(imgs[i]).save(os.path.join(pub, "train/images", c["id"] + ".png"))
            Image.fromarray(masks[i] * 255).save(os.path.join(pub, "train/masks", c["id"] + ".png"))
            train_rows.append({"image_id": c["id"], "diagnosis": c["cls"]})

    train_df = pd.DataFrame(train_rows).sort_values("image_id")
    test_df = pd.DataFrame(test_rows).sort_values("image_id")
    train_df.to_csv(os.path.join(pub, "train_metadata.csv"), index=False)
    test_df[["image_id"]].to_csv(os.path.join(pub, "test_ids.csv"), index=False)
    pd.DataFrame({"image_id": test_df["image_id"], "rle": ""}).to_csv(
        os.path.join(pub, "sample_submission.csv"), index=False)

    order = np.argsort(gt_ids)
    np.savez_compressed(
        os.path.join(priv, "test_ground_truth.npz"),
        image_ids=np.array(gt_ids)[order],
        diagnosis=np.array(gt_cls)[order],
        masks_packed=np.packbits(np.stack(gt_masks)[order].reshape(len(gt_ids), -1), axis=1),
        image_size=np.array([IMAGE_SIZE]),
    )
    test_df.to_csv(os.path.join(priv, "test_mapping.csv"), index=False)

    print("\nSplit summary")
    both = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")], ignore_index=True)
    print(pd.crosstab(both["diagnosis"], both["split"], margins=True))
    print(f"\nModification: {args.modification} | test shift: {args.test_shift}")
    with open(os.path.join(priv, "build_info.json"), "w", encoding="utf-8") as f:
        json.dump({"modification": args.modification, "test_shift": args.test_shift,
                   "dup_hamming": args.dup_hamming, "test_frac": args.test_frac,
                   "n_train": len(train_df), "n_test": len(test_df)}, f, indent=2)
    print(f"\nPublic  -> {os.path.relpath(pub, ROOT)}/")
    print(f"Private -> {os.path.relpath(priv, ROOT)}/  (git-ignored)")
    print("Next: python scripts/encrypt_ground_truth.py")


if __name__ == "__main__":
    main()
