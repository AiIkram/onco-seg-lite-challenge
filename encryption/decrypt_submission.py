#!/usr/bin/env python3
"""
ORGANISER ONLY - decrypt a submission for manual inspection / auditing.
(CI never writes decrypted files to disk; see scripts/ci_score.py.)

    python encryption/decrypt_submission.py submissions/encrypted/alice.enc encryption/private_key.pem [out.csv]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_utils import decrypt_bytes, load_private_key  # noqa: E402

if len(sys.argv) not in (3, 4):
    sys.exit("Usage: python encryption/decrypt_submission.py <file.enc> <private_key.pem> [out.csv]")

enc_path, key_path = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) == 4 else enc_path[:-4] + "_decrypted.csv"

with open(key_path, "rb") as f:
    key = load_private_key(f.read())
with open(enc_path, "rb") as f:
    plaintext = decrypt_bytes(f.read(), key)
with open(out_path, "wb") as f:
    f.write(plaintext)
print(f"Decrypted -> {out_path}  (do not commit)")
