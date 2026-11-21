#!/usr/bin/env python3
"""
ORGANISER - score a plain-text submission against the private ground truth.

    python competition/evaluate.py --submission sub.csv [--ground_truth data/private/test_ground_truth.npz]

Prints `SCORE: <mean dice>` plus per-class breakdown. Useful for scoring baselines,
LLM runs or late submissions offline.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from competition.metrics import (evaluate_masks, read_submission_csv,  # noqa: E402
                                 submission_to_masks, validate_submission)


def load_gt(path):
    npz = np.load(path, allow_pickle=False)
    size = int(npz["image_size"][0])
    masks = np.unpackbits(npz["masks_packed"], axis=1)[:, : size * size].reshape(-1, size, size)
    ids = [str(i) for i in npz["image_ids"]]
    return dict(zip(ids, masks)), dict(zip(ids, map(str, npz["diagnosis"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--ground_truth", default=os.path.join(ROOT, "data", "private", "test_ground_truth.npz"))
    args = ap.parse_args()

    gt, cls = load_gt(args.ground_truth)
    sub = read_submission_csv(args.submission)
    errors = validate_submission(sub, gt.keys())
    if errors:
        print("INVALID:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    pred = submission_to_masks(sub)
    m = evaluate_masks(pred, gt)
    for k, v in m.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
    for c in sorted(set(cls.values())):
        ids = [i for i in gt if cls[i] == c]
        s = evaluate_masks({i: pred[i] for i in ids}, {i: gt[i] for i in ids})["score"]
        print(f"{'dice_' + c:>12}: {s:.4f}  (n={len(ids)})")
    print(f"SCORE: {m['score']:.4f}")


if __name__ == "__main__":
    main()
