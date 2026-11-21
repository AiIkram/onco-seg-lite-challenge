#!/usr/bin/env python3
"""
ORGANISER - score the official baseline offline (plaintext ground truth) and add it to
docs/leaderboard.csv as `organiser-baseline` (model_type `baseline`, excluded from Human Top).

    python scripts/add_baseline.py --submission submissions/organiser-baseline.csv
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from competition.evaluate import load_gt  # noqa: E402
from competition.metrics import evaluate_masks, read_submission_csv, submission_to_masks, validate_submission  # noqa: E402

FIELDS = ["team", "score", "iou", "lesion_dice", "specificity", "model_type",
          "architecture", "params", "llm", "date", "run_id"]

ap = argparse.ArgumentParser()
ap.add_argument("--submission", default=os.path.join(ROOT, "submissions", "organiser-baseline.csv"))
ap.add_argument("--metadata", default=os.path.join(ROOT, "submissions", "metadata", "organiser-baseline.json"))
a = ap.parse_args()

gt, _ = load_gt(os.path.join(ROOT, "data", "private", "test_ground_truth.npz"))
sub = read_submission_csv(a.submission)
err = validate_submission(sub, gt.keys())
if err:
    sys.exit(f"Baseline submission invalid: {err}")
m = evaluate_masks(submission_to_masks(sub), gt)
meta = json.load(open(a.metadata, encoding="utf-8")) if os.path.exists(a.metadata) else {}

lb = os.path.join(ROOT, "docs", "leaderboard.csv")
with open(lb, newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r.get("team") and r["team"] != "organiser-baseline"]
rows.insert(0, {
    "team": "organiser-baseline", "score": f"{m['score']:.4f}", "iou": f"{m['iou']:.4f}",
    "lesion_dice": f"{m['lesion_dice']:.4f}", "specificity": f"{m['specificity']:.4f}",
    "model_type": "baseline", "architecture": meta.get("architecture", "TinyUNet-b16"),
    "params": str(meta.get("params", "")), "llm": "",
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "run_id": "baseline",
})
with open(lb, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(f"Baseline mean Dice on hidden test: {m['score']:.4f} (IoU {m['iou']:.4f}, "
      f"lesion Dice {m['lesion_dice']:.4f}, specificity {m['specificity']:.4f}) -> docs/leaderboard.csv")
