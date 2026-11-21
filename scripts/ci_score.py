#!/usr/bin/env python3
"""
CI ONLY - called by .github/workflows/score_submission.yml

Decrypts the submission and the hidden ground truth IN MEMORY (nothing decrypted
is written to disk), validates, scores, appends to docs/leaderboard.csv and writes
a PR comment to --comment-out. Status goes to $GITHUB_OUTPUT as `status=`:
    ok | invalid | decrypt_error | limit_reached | closed

    PRIVATE_KEY="$(cat private_key.pem)" python scripts/ci_score.py \
        --submission /tmp/sub.enc --name alice --user alice-gh --pr 12 \
        [--metadata /tmp/meta.json] [--max-submissions 1] --comment-out /tmp/comment.md
"""
import argparse
import csv
import io
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "encryption"))

from competition.metrics import (IMAGE_SIZE, evaluate_masks, read_submission_csv,  # noqa: E402
                                 submission_to_masks, validate_submission)
from crypto_utils import decrypt_bytes, load_private_key  # noqa: E402

LEADERBOARD = os.path.join(ROOT, "docs", "leaderboard.csv")
GT_ENC = os.path.join(ROOT, "data", "private_encrypted", "test_ground_truth.enc")
FIELDS = ["team", "score", "iou", "lesion_dice", "specificity", "model_type",
          "architecture", "params", "llm", "date", "run_id"]
MODEL_TYPES = {"human", "llm-only", "human+llm"}
PARAM_LIMIT = 5_000_000


def set_output(**kv):
    path = os.environ.get("GITHUB_OUTPUT")
    for k, v in kv.items():
        print(f"[output] {k}={v}")
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{k}={v}\n")


def clean(s, maxlen=60):
    s = re.sub(r"[\r\n,|<>`]", " ", str(s)).strip()
    return s[:maxlen]


def read_leaderboard():
    if not os.path.exists(LEADERBOARD):
        return []
    with open(LEADERBOARD, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("team")]


def load_metadata(path):
    meta = {"model_type": "unspecified", "architecture": "", "params": "", "llm": ""}
    warnings = []
    if not path or not os.path.exists(path):
        warnings.append("No metadata JSON found - model type recorded as `unspecified`.")
        return meta, warnings
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("metadata must be a JSON object")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Metadata JSON could not be parsed ({clean(e, 120)}) - ignored.")
        return meta, warnings

    mt = str(raw.get("model_type", "")).strip().lower()
    if mt in MODEL_TYPES:
        meta["model_type"] = mt
    else:
        warnings.append(f"`model_type` must be one of {sorted(MODEL_TYPES)} - recorded as `unspecified`.")
    meta["architecture"] = clean(raw.get("architecture", ""))
    meta["llm"] = clean(raw.get("llm", ""))
    try:
        p = int(raw.get("params", ""))
        meta["params"] = str(p)
        if p > PARAM_LIMIT:
            warnings.append(f"Declared {p:,} parameters > {PARAM_LIMIT:,} limit - flagged for organiser review.")
    except (TypeError, ValueError):
        warnings.append("`params` missing or not an integer.")
    try:
        n_runs = int(raw.get("n_tuning_runs", 0))
        if n_runs > 5:
            warnings.append(f"Declared {n_runs} tuning runs > 5 allowed by the protocol - flagged for review.")
    except (TypeError, ValueError):
        pass
    try:
        mins = float(raw.get("train_minutes_total") or 0)
        if mins > 180:
            warnings.append(f"Declared {mins:.0f} training minutes > 180 CPU-min budget - flagged for review.")
    except (TypeError, ValueError):
        pass
    return meta, warnings


def load_ground_truth(private_key):
    with open(GT_ENC, "rb") as f:
        npz = np.load(io.BytesIO(decrypt_bytes(f.read(), private_key)), allow_pickle=False)
    ids = [str(x) for x in npz["image_ids"]]
    size = int(npz["image_size"][0])
    assert size == IMAGE_SIZE, "ground truth image size mismatch"
    masks = np.unpackbits(npz["masks_packed"], axis=1)[:, : size * size].reshape(-1, size, size)
    return ({i: m for i, m in zip(ids, masks)},
            {i: str(c) for i, c in zip(ids, npz["diagnosis"])})


def rank_of(rows, team):
    best = {}
    for r in rows:
        try:
            s = float(r["score"])
        except ValueError:
            continue
        best[r["team"]] = max(best.get(r["team"], -1.0), s)
    if team not in best:
        return None, len(best)
    return 1 + sum(1 for s in best.values() if s > best[team] + 1e-12), len(best)


def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--name", required=True, help="submission file stem")
    ap.add_argument("--user", required=True, help="GitHub login of the PR author")
    ap.add_argument("--pr", required=True)
    ap.add_argument("--metadata")
    ap.add_argument("--has-report", default="", help="non-empty if the PR includes submissions/reports/<name>.md")
    ap.add_argument("--max-submissions", type=int, default=1)
    ap.add_argument("--deadline", default="", help="ISO-8601 UTC; submissions after it are not scored")
    ap.add_argument("--comment-out", required=True)
    args = ap.parse_args()

    user = clean(args.user, 39)
    lines = []

    def finish(status):
        with open(args.comment_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        set_output(status=status)
        sys.exit(0)

    # 0) deadline
    if args.deadline:
        dl = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > dl:
            lines += ["## ⏰ Competition closed",
                      f"The submission deadline ({dl:%Y-%m-%d %H:%M} UTC) has passed. This PR was not scored."]
            finish("closed")

    # 1) submission limit
    rows = read_leaderboard()
    used = sum(1 for r in rows if r["team"] == user)
    if used >= args.max_submissions:
        lines += ["## ⛔ Submission limit reached",
                  f"@{user} already has **{used}/{args.max_submissions}** scored submission(s). "
                  "This PR was not scored and does not count."]
        finish("limit_reached")

    pem = os.environ.get("PRIVATE_KEY", "")
    if "BEGIN" not in pem:
        print("PRIVATE_KEY secret missing or malformed", file=sys.stderr)
        sys.exit(2)  # organiser problem -> red CI, no comment about the participant
    private_key = load_private_key(pem.encode())

    # 2) decrypt submission (in memory)
    try:
        with open(args.submission, "rb") as f:
            plaintext = decrypt_bytes(f.read(), private_key)
        sub = read_submission_csv(io.BytesIO(plaintext))
    except Exception as e:  # noqa: BLE001
        lines += ["## ❌ Could not decrypt your file",
                  "This attempt does **not** count.",
                  "Re-create it with the official script and push again:",
                  "```bash\npython encryption/encrypt_submission.py submissions/<your_name>.csv\n```",
                  f"<sub>detail: {clean(e, 160)}</sub>"]
        finish("decrypt_error")

    # 3) validate
    gt_masks, gt_cls = load_ground_truth(private_key)
    errors = validate_submission(sub, gt_masks.keys())
    if errors:
        lines += ["## ❌ Invalid submission format", "This attempt does **not** count.", ""]
        lines += [f"- {clean(e, 200)}" for e in errors]
        lines += ["", "Required: `image_id,rle` with one row per id in `data/public/test_ids.csv`. "
                  "Run `python starter_code/check_submission.py <file.csv>` locally."]
        finish("invalid")

    # 4) score
    pred = submission_to_masks(sub)
    m = evaluate_masks(pred, gt_masks)
    per_cls = {}
    for cls in ("benign", "malignant", "normal"):
        ids = [i for i, c in gt_cls.items() if c == cls]
        if ids:
            per_cls[cls] = evaluate_masks({i: pred[i] for i in ids}, {i: gt_masks[i] for i in ids})["score"]

    meta, warnings = load_metadata(args.metadata)
    if not args.has_report:
        warnings.append("No documentation log (`submissions/reports/<name>.md`) in this PR - "
                        "required by the protocol; post it as a comment on this PR.")

    # 5) append to leaderboard
    row = {
        "team": user, "score": f"{m['score']:.4f}", "iou": f"{m['iou']:.4f}",
        "lesion_dice": fmt(m["lesion_dice"]), "specificity": fmt(m["specificity"]),
        "model_type": meta["model_type"], "architecture": meta["architecture"],
        "params": meta["params"], "llm": meta["llm"],
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "run_id": clean(f"pr{args.pr}-{args.name}", 80),
    }
    os.makedirs(os.path.dirname(LEADERBOARD), exist_ok=True)
    new_file = not os.path.exists(LEADERBOARD) or os.path.getsize(LEADERBOARD) == 0
    with open(LEADERBOARD, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)

    rank, n_teams = rank_of(read_leaderboard(), user)

    lines += [
        "## 🎯 Submission scored",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Mean Dice (leaderboard score)** | **`{m['score']:.4f}`** |",
        f"| Mean IoU | `{m['iou']:.4f}` |",
        f"| Dice on lesion images only | `{fmt(m['lesion_dice'])}` |",
        f"| Specificity on normal scans (no false blobs) | `{fmt(m['specificity'])}` |",
        f"| Lesion detection sensitivity | `{fmt(m['sensitivity'])}` |",
    ]
    for cls, s in per_cls.items():
        lines.append(f"| Dice - {cls} | `{s:.4f}` |")
    lines += [
        "",
        f"🏆 **Current rank: #{rank} of {n_teams}** (best submission per participant)",
        f"📨 Submissions used: **{used + 1}/{args.max_submissions}**",
        f"🧾 Recorded as: `{meta['model_type']}` {('· ' + meta['architecture']) if meta['architecture'] else ''}",
    ]
    if warnings:
        lines += ["", "**Notes**"] + [f"- {w}" for w in warnings]
    lines += ["", "🔒 Your predictions and the test masks were decrypted in memory only and never written to disk."]
    set_output(score=row["score"], rank=rank)
    finish("ok")


if __name__ == "__main__":
    main()
