#!/usr/bin/env python3
"""
ORGANISER ONLY - encrypt the hidden test masks so CI can score without them ever
being readable in the repository.

    python scripts/encrypt_ground_truth.py

Reads   data/private/test_ground_truth.npz         (git-ignored)
Writes  data/private_encrypted/test_ground_truth.enc (commit this)

Why not a GitHub secret like the GNN template? Secrets are capped at 48 KB and
segmentation masks are bigger. The masks are encrypted with the same public key
as submissions; only the PRIVATE_KEY secret (inside CI) can open them.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "encryption"))
from crypto_utils import encrypt_bytes, load_public_key  # noqa: E402

src = os.path.join(ROOT, "data", "private", "test_ground_truth.npz")
dst = os.path.join(ROOT, "data", "private_encrypted", "test_ground_truth.enc")
pub = os.path.join(ROOT, "encryption", "public_key.pem")

for p, hint in ((src, "run scripts/prepare_data.py first"), (pub, "run encryption/generate_keys.py first")):
    if not os.path.exists(p):
        sys.exit(f"Missing {os.path.relpath(p, ROOT)} - {hint}")

with open(pub, "rb") as f:
    key = load_public_key(f.read())
with open(src, "rb") as f:
    blob = encrypt_bytes(f.read(), key)
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "wb") as f:
    f.write(blob)
print(f"Encrypted ground truth -> {os.path.relpath(dst, ROOT)} ({len(blob):,} bytes). Commit this file.")
