#!/usr/bin/env python3
"""
Validate a submission CSV locally (format only - the score is computed in CI).

    python starter_code/check_submission.py submissions/alice.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from competition.metrics import read_submission_csv, submission_to_masks, validate_submission  # noqa: E402

if len(sys.argv) != 2:
    sys.exit("Usage: python starter_code/check_submission.py <submission.csv>")

ids = pd.read_csv(os.path.join(ROOT, "data", "public", "test_ids.csv"), dtype=str).image_id
sub = read_submission_csv(sys.argv[1])
errors = validate_submission(sub, ids)
if errors:
    print("INVALID")
    for e in errors:
        print("  -", e)
    sys.exit(1)

masks = submission_to_masks(sub)
areas = np.array([m.sum() for m in masks.values()])
print("VALID - format OK")
print(f"  rows: {len(sub)}")
print(f"  images with a predicted lesion: {(areas > 0).sum()} / {len(areas)}")
if (areas > 0).any():
    print(f"  lesion area (px, non-empty): median {int(np.median(areas[areas > 0]))}, "
          f"min {areas[areas > 0].min()}, max {areas.max()}")
print("Next: python encryption/encrypt_submission.py", sys.argv[1])
