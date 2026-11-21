#!/usr/bin/env python3
"""
PARTICIPANTS - encrypt your predictions before opening a Pull Request.

    python encryption/encrypt_submission.py submissions/<your_name>.csv

The CSV is validated locally first (format only - no score), so a broken file
never costs you a submission. Output: submissions/encrypted/<your_name>.enc
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "encryption"))

import pandas as pd  # noqa: E402
from competition.metrics import read_submission_csv, validate_submission  # noqa: E402
from crypto_utils import encrypt_bytes, load_public_key  # noqa: E402

NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{2,60}$")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python encryption/encrypt_submission.py submissions/<your_name>.csv")
    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        sys.exit(f"File not found: {csv_path}")

    name = os.path.splitext(os.path.basename(csv_path))[0]
    if not NAME_RE.match(name):
        sys.exit(f"File name '{name}' must be 2-60 chars of letters, digits, '_' or '-'.")

    # 1) local format check
    test_ids = pd.read_csv(os.path.join(ROOT, "data", "public", "test_ids.csv"), dtype=str)["image_id"]
    sub = read_submission_csv(csv_path)
    errors = validate_submission(sub, test_ids)
    if errors:
        print("Submission is INVALID - nothing was encrypted:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"Format OK: {len(sub)} rows, {int((sub['rle'] != '').sum())} non-empty masks")

    # 2) encrypt
    pub_path = os.path.join(ROOT, "encryption", "public_key.pem")
    if not os.path.exists(pub_path):
        sys.exit("encryption/public_key.pem is missing - pull the latest version of the repository.")
    with open(pub_path, "rb") as f:
        public_key = load_public_key(f.read())
    with open(csv_path, "rb") as f:
        blob = encrypt_bytes(f.read(), public_key)

    out_dir = os.path.join(ROOT, "submissions", "encrypted")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.enc")
    with open(out_path, "wb") as f:
        f.write(blob)

    rel = os.path.relpath(out_path, ROOT)
    print(f"Encrypted -> {rel} ({len(blob):,} bytes)")
    print("\nNext steps:")
    print(f"  1. (optional) add submissions/metadata/{name}.json  - see README")
    print(f"  2. git add {rel}")
    print(f'  3. git commit -m "Submission: {name}" && git push')
    print("  4. Open a Pull Request -> the bot comments your score in ~2 minutes")
    print("\nDo NOT commit the plain CSV.")


if __name__ == "__main__":
    main()
