#!/usr/bin/env python3
"""
Fill the FROZEN prompt template (llm_eval/prompt_template.md) from the repository.
The template is identical across competitions; only the slot contents change
(GNN-CB 3.3.1 "Frozen Prompt").

Slots: {{README}} {{REPO_TREE}} {{DATASET_SUMMARY}} {{SUBMISSION_FORMAT}} {{ALLOWED_LIBRARIES}} {{TIME_BUDGET_MIN}}

    python llm_eval/build_prompt.py --out llm_eval/prompt.md
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from competition import metrics  # noqa: E402

TEMPLATE = os.path.join(ROOT, "llm_eval", "prompt_template.md")
SKIP_DIRS = {".git", "__pycache__", "private", "checkpoints", "runs", "llm_eval", ".github", "encryption"}


PUBLIC_ITEMS = ["README.md", "RULES.md", "LICENSE", "competition", "starter_code", "data/public"]


def repo_tree(root=ROOT, max_files_per_dir=8):
    """Tree of exactly what the LLM can see (the public items copied into the sandbox)."""
    lines = []
    for item in PUBLIC_ITEMS:
        top = os.path.join(root, item)
        if os.path.isfile(top):
            lines.append(item)
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            rel_top = os.path.relpath(dirpath, top)
            depth = 0 if rel_top == "." else rel_top.count(os.sep) + 1
            lines.append(item + "/" if depth == 0 else "    " * depth + os.path.basename(dirpath) + "/")
            files = sorted(f for f in filenames if not f.startswith(".") and f != "runs_log.csv"
                           and not f.endswith((".enc", ".pt")))
            pngs = [f for f in files if f.endswith(".png")]
            others = [f for f in files if not f.endswith(".png")]
            for f in others[:max_files_per_dir]:
                lines.append("    " * (depth + 1) + f)
            if len(others) > max_files_per_dir:
                lines.append("    " * (depth + 1) + f"... ({len(others) - max_files_per_dir} more files)")
            if pngs:
                lines.append("    " * (depth + 1) + f"<{len(pngs)} PNG files, e.g. {pngs[0]}>")
    return "\n".join(lines)


def dataset_summary():
    pub = os.path.join(ROOT, "data", "public")
    meta = pd.read_csv(os.path.join(pub, "train_metadata.csv"), dtype=str)
    test_ids = pd.read_csv(os.path.join(pub, "test_ids.csv"), dtype=str)
    sample_img = Image.open(os.path.join(pub, "train", "images", meta.image_id.iloc[0] + ".png"))
    cover = []
    for i in meta.image_id:
        m = np.asarray(Image.open(os.path.join(pub, "train", "masks", i + ".png"))) > 0
        cover.append(m.mean())
    cover = np.array(cover)
    counts = meta.diagnosis.value_counts().to_dict()
    return "\n".join([
        f"- Training images: {len(meta)} (with masks); test images: {len(test_ids)} (no masks).",
        f"- Image format: PNG, mode {sample_img.mode}, size {sample_img.size[0]}x{sample_img.size[1]}.",
        "- Mask format: PNG, 0 = background, 255 = lesion.",
        f"- Training diagnosis counts: {counts} (diagnosis is available for training images only).",
        f"- Fraction of training images with an empty mask: {np.mean(cover == 0):.3f}.",
        f"- Lesion area as fraction of the image (non-empty masks): median {np.median(cover[cover > 0]):.3f}, "
        f"min {cover[cover > 0].min():.4f}, max {cover.max():.3f}.",
        "- Files: data/public/train_metadata.csv (image_id, diagnosis), data/public/test_ids.csv (image_id), "
        "data/public/sample_submission.csv.",
    ])


def submission_format():
    doc = metrics.__doc__.strip()
    return doc + "\n\nHelper functions available in `competition/metrics.py`: rle_encode(mask), rle_decode(rle), " \
                 "validate_submission(df, expected_ids), evaluate_masks(pred, gt)."


def allowed_libraries():
    with open(os.path.join(ROOT, "starter_code", "requirements.txt"), encoding="utf-8") as f:
        reqs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return "Python standard library and: " + ", ".join(reqs) + ". Nothing else is installed."


def build(time_budget_min=180):
    with open(TEMPLATE, encoding="utf-8") as f:
        tpl = f.read()
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    slots = {
        "{{README}}": readme,
        "{{REPO_TREE}}": repo_tree(),
        "{{DATASET_SUMMARY}}": dataset_summary(),
        "{{SUBMISSION_FORMAT}}": submission_format(),
        "{{ALLOWED_LIBRARIES}}": allowed_libraries(),
        "{{TIME_BUDGET_MIN}}": str(time_budget_min),
    }
    for k, v in slots.items():
        tpl = tpl.replace(k, v)
    return tpl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "llm_eval", "prompt.md"))
    ap.add_argument("--time-budget-min", type=int, default=180)
    a = ap.parse_args()
    text = build(a.time_budget_min)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Prompt written to {os.path.relpath(a.out, ROOT)} ({len(text):,} chars, ~{len(text) // 4:,} tokens)")
