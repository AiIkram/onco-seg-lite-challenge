#!/usr/bin/env python3
"""
Predict the test set and write, for submission <name>:
    submissions/<name>.csv               predictions (NEVER commit - encrypt it)
    submissions/metadata/<name>.json     model type, architecture, params, budget
    submissions/reports/<name>.md        documentation log, pre-filled from runs_log.csv - complete it!

    cd starter_code
    python generate_submission.py --name alice            # uses the best run in runs_log.csv
    python generate_submission.py --name alice --ckpt checkpoints/run3.pt

Then, from the repository root:
    python encryption/encrypt_submission.py submissions/alice.csv
"""
import argparse
import csv
import json
import os
import platform
import time

import pandas as pd
import torch

from common import ROOT, TinyUNet, count_params, get_device, load_test, postprocess, predict_probs
from competition.metrics import rle_encode, validate_submission

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_LOG = os.path.join(HERE, "runs_log.csv")


def load_runs():
    if not os.path.exists(RUNS_LOG):
        return []
    with open(RUNS_LOG, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="letters, digits, _ or - (becomes the file name)")
    ap.add_argument("--ckpt", default=None, help="default: best run (val Dice) in runs_log.csv")
    ap.add_argument("--model-type", default="human", choices=["human", "llm-only", "human+llm", "baseline"])
    ap.add_argument("--llm", default="", help="LLM name if one was used")
    args = ap.parse_args()

    runs = load_runs()
    ckpt_path = args.ckpt
    if ckpt_path is None:
        if not runs:
            raise SystemExit("No runs_log.csv found - train first or pass --ckpt.")
        best = max(runs, key=lambda r: float(r["val_dice"]))
        ckpt_path = os.path.join(HERE, "checkpoints", f"{best['run']}.pt")
        print(f"Using best run '{best['run']}' (val Dice {best['val_dice']})")

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TinyUNet(base=ckpt["base"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    t0 = time.time()
    ids, imgs = load_test()
    probs = predict_probs(model, imgs, device, size=ckpt["img_size"], tta=True)
    thr, min_area = ckpt.get("threshold", 0.5), ckpt.get("min_area", 0)
    rles = [rle_encode(postprocess(p, thr, min_area)) for p in probs]
    sub = pd.DataFrame({"image_id": ids, "rle": rles})
    errors = validate_submission(sub, ids)
    assert not errors, errors
    infer_s = time.time() - t0
    print(f"Inference: {len(ids)} images in {infer_s:.1f}s on {device} "
          f"({sum(r != '' for r in rles)} with a predicted lesion)")

    out_dir = os.path.join(ROOT, "submissions")
    for d in ("metadata", "reports"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    csv_path = os.path.join(out_dir, f"{args.name}.csv")
    sub.to_csv(csv_path, index=False)

    total_min = sum(float(r["minutes"]) for r in runs) if runs else None
    hardware = f"{platform.processor() or platform.machine()} | {os.cpu_count()} threads | {device}"
    meta = {
        "model_type": args.model_type,
        "architecture": f"TinyUNet-b{ckpt['base']}",
        "params": count_params(model),
        "llm": args.llm,
        "seed": ckpt.get("seed", 25),
        "n_tuning_runs": len(runs),
        "train_minutes_total": round(total_min, 1) if total_min is not None else None,
        "hardware": hardware,
    }
    meta_path = os.path.join(out_dir, "metadata", f"{args.name}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report_path = os.path.join(out_dir, "reports", f"{args.name}.md")
    if not os.path.exists(report_path):
        rows = "\n".join(
            f"| {r['run']} | {r['img_size']} | {r['base']} | {r['lr']} | {r['batch_size']} | {r['epochs_run']} "
            f"| {r['best_epoch']} | {r['minutes']} | {r['val_dice']} |" for r in runs) or "| - |"
        with open(os.path.join(ROOT, "submissions", "REPORT_TEMPLATE.md"), encoding="utf-8") as f:
            tpl = f.read()
        report = (tpl.replace("{{NAME}}", args.name)
                     .replace("{{MODEL_TYPE}}", args.model_type)
                     .replace("{{ARCH}}", meta["architecture"])
                     .replace("{{PARAMS}}", f"{meta['params']:,}")
                     .replace("{{HARDWARE}}", hardware)
                     .replace("{{TOTAL_MIN}}", str(meta["train_minutes_total"]))
                     .replace("{{INFER_S}}", f"{infer_s:.1f}")
                     .replace("{{N_RUNS}}", str(len(runs)))
                     .replace("{{RUN_ROWS}}", rows))
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

    print(f"Wrote {os.path.relpath(csv_path, ROOT)}, {os.path.relpath(meta_path, ROOT)}, "
          f"{os.path.relpath(report_path, ROOT)}")
    print("Complete the report (search strategy, loss-curve check, LLM usage), then from the repo root:")
    print(f"  python encryption/encrypt_submission.py submissions/{args.name}.csv")


if __name__ == "__main__":
    main()
