"""
ONCO-SEG Lite Challenge - official metrics, RLE codec and submission validation.

This file is the single source of truth for scoring. Participants may import it
locally (e.g. to validate on their own hold-out split) but must not modify it.

Mask convention
---------------
* Every mask is a binary array of shape (IMAGE_SIZE, IMAGE_SIZE) = (256, 256).
* 1 = tumour / lesion pixel, 0 = background.

RLE convention (Kaggle-style, row-major)
----------------------------------------
* Flatten the mask in row-major (C) order: pixel (r, c) -> index r * 256 + c.
* Encode runs of 1s as space-separated pairs "start length".
* `start` is 1-indexed. Runs must be sorted and must not overlap.
* An empty mask (no lesion) is an EMPTY string.

    mask = [[0, 1, 1],
            [0, 0, 1]]   ->  "2 2 6 1"

Metric
------
Primary score = mean per-image Dice similarity coefficient (DSC).
    * both prediction and ground truth empty  -> Dice = 1.0 (correct "no lesion")
    * exactly one of them empty               -> Dice = 0.0
    * otherwise                               -> 2|P∩G| / (|P| + |G|)

Because ~17% of test images are normal (no lesion), false-positive blobs on
normal scans are penalised as hard as missing a tumour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IMAGE_SIZE = 256
N_PIXELS = IMAGE_SIZE * IMAGE_SIZE
REQUIRED_COLUMNS = ("image_id", "rle")


# --------------------------------------------------------------------------- #
# RLE                                                                          #
# --------------------------------------------------------------------------- #
def rle_encode(mask: np.ndarray) -> str:
    """Binary mask (H, W) -> RLE string (row-major, 1-indexed)."""
    flat = (np.asarray(mask).reshape(-1) > 0).astype(np.uint8)
    if flat.sum() == 0:
        return ""
    padded = np.concatenate([[0], flat, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1  # 1-indexed
    starts, ends = changes[0::2], changes[1::2]
    lengths = ends - starts
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


def rle_decode(rle, shape=(IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    """RLE string -> binary uint8 mask. Raises ValueError on malformed input."""
    n = shape[0] * shape[1]
    mask = np.zeros(n, dtype=np.uint8)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return mask.reshape(shape)
    rle = str(rle).strip()
    if rle == "":
        return mask.reshape(shape)

    tokens = rle.split()
    if len(tokens) % 2 != 0:
        raise ValueError("RLE must contain an even number of integers (start length pairs)")
    try:
        nums = np.array([int(t) for t in tokens], dtype=np.int64)
    except ValueError:
        raise ValueError("RLE contains non-integer tokens")

    starts, lengths = nums[0::2], nums[1::2]
    if (starts < 1).any():
        raise ValueError("RLE starts must be >= 1 (1-indexed)")
    if (lengths < 1).any():
        raise ValueError("RLE run lengths must be >= 1")
    ends = starts - 1 + lengths  # exclusive, 0-indexed
    if (ends > n).any():
        raise ValueError(f"RLE run exceeds image bounds ({n} pixels)")
    if len(starts) > 1:
        if (np.diff(starts) <= 0).any():
            raise ValueError("RLE runs must be sorted by start position")
        if (starts[1:] - 1 < ends[:-1]).any():
            raise ValueError("RLE runs must not overlap")

    for s, e in zip(starts - 1, ends):
        mask[s:e] = 1
    return mask.reshape(shape)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    ps, gs = p.sum(), g.sum()
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    return float(2.0 * np.logical_and(p, g).sum() / (ps + gs))


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def evaluate_masks(pred_masks: dict, gt_masks: dict) -> dict:
    """
    pred_masks / gt_masks: {image_id: (256,256) binary array}
    Returns a dict of aggregate metrics. `score` is the leaderboard metric.
    """
    dices, ious = [], []
    lesion_dice = []
    tn = fp = tp = fn = 0  # image-level "is there a lesion?" detection
    for image_id, gt in gt_masks.items():
        pred = pred_masks[image_id]
        d = dice_score(pred, gt)
        dices.append(d)
        ious.append(iou_score(pred, gt))
        has_gt, has_pred = gt.any(), pred.any()
        if has_gt:
            lesion_dice.append(d)
            tp += int(has_pred)
            fn += int(not has_pred)
        else:
            tn += int(not has_pred)
            fp += int(has_pred)

    return {
        "score": float(np.mean(dices)),                      # mean Dice (primary)
        "iou": float(np.mean(ious)),
        "lesion_dice": float(np.mean(lesion_dice)) if lesion_dice else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),  # on normal scans
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),  # lesion detected at all
        "n_images": len(dices),
    }


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate_submission(sub: pd.DataFrame, expected_ids) -> list:
    """Return a list of human-readable errors (empty list == valid)."""
    errors = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in sub.columns]
    if missing_cols:
        return [f"Missing column(s) {missing_cols}. Expected header: image_id,rle"]

    expected = set(map(str, expected_ids))
    ids = sub["image_id"].astype(str)

    if len(sub) != len(expected):
        errors.append(f"Expected {len(expected)} rows, got {len(sub)}")
    dups = ids[ids.duplicated()].unique().tolist()
    if dups:
        errors.append(f"Duplicate image_id(s): {dups[:5]}{' ...' if len(dups) > 5 else ''}")
    missing = expected - set(ids)
    extra = set(ids) - expected
    if missing:
        errors.append(f"{len(missing)} test image_id(s) missing, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} unknown image_id(s), e.g. {sorted(extra)[:3]}")

    bad = 0
    first_err = None
    for image_id, rle in zip(ids, sub["rle"]):
        try:
            rle_decode(rle)
        except ValueError as e:
            bad += 1
            first_err = first_err or f"{image_id}: {e}"
    if bad:
        errors.append(f"{bad} malformed RLE string(s). First: {first_err}")
    return errors


def submission_to_masks(sub: pd.DataFrame) -> dict:
    return {str(i): rle_decode(r) for i, r in zip(sub["image_id"].astype(str), sub["rle"])}


def read_submission_csv(path_or_buffer) -> pd.DataFrame:
    """Read a submission keeping empty RLE strings as '' (not NaN) and ids as str."""
    return pd.read_csv(path_or_buffer, dtype={"image_id": str, "rle": str},
                       keep_default_na=False)
