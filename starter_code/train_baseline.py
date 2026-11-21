#!/usr/bin/env python3
"""
Baseline: tiny U-Net (~1.9M params), BCE + Dice loss, flips + gain jitter.
CPU-first, following the GNN-CB human protocol (seed 25, 3 CPU-hour budget,
<= 5 tuning runs, loss curves for convergence checks, validation-only selection).

    cd starter_code
    python train_baseline.py                     # CPU, 128 px: ~20-40 min on a laptop
    python train_baseline.py --img-size 256      # sharper masks, ~3-4x slower on CPU

Every run:
  * holds out 15% of train (stratified by diagnosis) - the ONLY data used for model selection
  * keeps the checkpoint with the best OFFICIAL metric, tunes threshold/min-area on validation
  * saves checkpoints/<run>_loss_curve.png and appends one line to runs_log.csv
    (counts your tuning runs; the protocol allows at most 5)
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from common import (TinyUNet, SegDataset, bce_dice_loss, count_params, get_device,
                    load_train, postprocess, predict_probs, seed_everything)
from competition.metrics import evaluate_masks

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_LOG = os.path.join(HERE, "runs_log.csv")
MAX_TUNING_RUNS = 5


def split_stratified(meta, frac, seed):
    rng = np.random.default_rng(seed)
    val = []
    for _, grp in meta.groupby("diagnosis"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        val.extend(idx[: max(1, round(frac * len(idx)))])
    val = np.array(sorted(val))
    tr = np.setdiff1d(np.arange(len(meta)), val)
    return tr, val


def official_score(probs, gts, thr=0.5, min_area=0):
    pred = {i: postprocess(p, thr, min_area) for i, p in enumerate(probs)}
    return evaluate_masks(pred, {i: g.astype(np.uint8) for i, g in enumerate(gts)})


def previous_runs():
    if not os.path.exists(RUNS_LOG):
        return 0
    with open(RUNS_LOG, encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def save_curves(history, path):
    """Loss + validation Dice curves (protocol: verify convergence)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        with open(path.replace(".png", ".json"), "w", encoding="utf-8") as f:
            json.dump(history, f)
        return path.replace(".png", ".json")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(history["epoch"], history["train_loss"], label="train loss", color="#0f766e")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE + Dice loss")
    ax2 = ax1.twinx()
    ax2.plot(history["val_epoch"], history["val_dice"], "o-", label="val mean Dice", color="#be185d")
    ax2.set_ylabel("val mean Dice")
    ax2.set_ylim(0, 1)
    fig.legend(loc="lower left", bbox_to_anchor=(0.1, 0.12), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=16, help="U-Net width")
    ap.add_argument("--img-size", type=int, default=128, help="train/infer size (output is always 256)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=25, help="protocol seed: 25")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--time-budget-min", type=float, default=150,
                    help="stop training after this many minutes (protocol total: 180 CPU-min for ALL runs)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default=None, help="checkpoint path (default checkpoints/<run-name>.pt)")
    args = ap.parse_args()

    n_prev = previous_runs()
    if n_prev >= MAX_TUNING_RUNS:
        print(f"WARNING: runs_log.csv already lists {n_prev} runs - the protocol allows at most "
              f"{MAX_TUNING_RUNS} tuning runs. Declare this in your report.")
    run_name = args.run_name or f"run{n_prev + 1}"
    out = args.out or os.path.join(HERE, "checkpoints", f"{run_name}.pt")

    seed_everything(args.seed)
    device = get_device()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    meta, imgs, masks = load_train()
    tr, va = split_stratified(meta, args.val_frac, args.seed)
    print(f"{run_name} | device={device} | train={len(tr)} val={len(va)} | size={args.img_size}")

    model = TinyUNet(base=args.base).to(device)
    n_params = count_params(model)
    print(f"TinyUNet params: {n_params:,}")
    assert n_params <= 5_000_000, "Challenge limit is 5M parameters"

    g = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        SegDataset(imgs[tr], masks[tr], size=args.img_size, augment=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    history = {"epoch": [], "train_loss": [], "val_epoch": [], "val_dice": []}
    best, best_epoch, t0 = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = bce_dice_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)))
        out_of_time = (time.time() - t0) / 60 > args.time_budget_min
        if epoch % 2 == 0 or epoch == args.epochs or out_of_time:
            probs = predict_probs(model, imgs[va], device, size=args.img_size, tta=False)
            score = official_score(probs, masks[va])["score"]
            history["val_epoch"].append(epoch)
            history["val_dice"].append(score)
            flag = ""
            if score > best:
                best, best_epoch, flag = score, epoch, "  * saved"
                torch.save({"state_dict": model.state_dict(), "base": args.base,
                            "img_size": args.img_size}, out)
            print(f"epoch {epoch:3d} | loss {np.mean(losses):.4f} | val Dice {score:.4f}{flag} "
                  f"| {time.time() - t0:.0f}s")
        if out_of_time:
            print(f"Time budget of {args.time_budget_min} min reached - stopping (early stop).")
            break

    # tune post-processing on validation with the best checkpoint
    ckpt = torch.load(out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    probs = predict_probs(model, imgs[va], device, size=args.img_size, tta=True)
    grid = [(t, a) for t in (0.3, 0.4, 0.5, 0.6) for a in (0, 50, 150, 300, 600)]
    results = {gp: official_score(probs, masks[va], *gp)["score"] for gp in grid}
    (thr, min_area), _ = max(results.items(), key=lambda kv: kv[1])
    final = official_score(probs, masks[va], thr, min_area)
    minutes = (time.time() - t0) / 60
    ckpt.update(threshold=thr, min_area=min_area, n_params=n_params, val=final, seed=args.seed)
    torch.save(ckpt, out)
    curve = save_curves(history, out.replace(".pt", "_loss_curve.png"))

    new_log = not os.path.exists(RUNS_LOG)
    with open(RUNS_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["run", "timestamp", "minutes", "device", "img_size", "base", "epochs_run",
                        "best_epoch", "lr", "batch_size", "seed", "params", "val_dice", "threshold", "min_area"])
        w.writerow([run_name, datetime.now().isoformat(timespec="seconds"), f"{minutes:.1f}", device,
                    args.img_size, args.base, history["epoch"][-1], best_epoch, args.lr, args.batch_size,
                    args.seed, n_params, f"{final['score']:.4f}", thr, min_area])

    print("\n" + "=" * 60)
    print(f"Best val mean Dice (TTA, thr={thr}, min_area={min_area}): {final['score']:.4f}")
    print(f"  IoU {final['iou']:.4f} | lesion Dice {final['lesion_dice']:.4f} | "
          f"specificity {final['specificity']:.4f}")
    print(f"Training time: {minutes:.1f} min on {device} | run {n_prev + 1}/{MAX_TUNING_RUNS}")
    print(f"Checkpoint: {os.path.relpath(out)} | curves: {os.path.relpath(curve)}")
    print(f"Next: python generate_submission.py --name <your_name> --ckpt {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
