#!/usr/bin/env python3
"""
ORGANISER ONLY - run once before launching the challenge.

    python encryption/generate_keys.py

Creates:
    encryption/public_key.pem   -> commit it (participants encrypt with it)
    encryption/private_key.pem  -> NEVER commit. Paste its content into the
                                   GitHub secret PRIVATE_KEY, then store it offline.
"""
import os
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

HERE = os.path.dirname(os.path.abspath(__file__))
priv_path = os.path.join(HERE, "private_key.pem")
pub_path = os.path.join(HERE, "public_key.pem")

if os.path.exists(priv_path) and "--force" not in sys.argv:
    sys.exit(f"{priv_path} already exists. Re-run with --force to overwrite "
             "(this invalidates every existing encrypted submission and the encrypted test masks).")

key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
with open(priv_path, "wb") as f:
    f.write(key.private_bytes(serialization.Encoding.PEM,
                              serialization.PrivateFormat.PKCS8,
                              serialization.NoEncryption()))
os.chmod(priv_path, 0o600)
with open(pub_path, "wb") as f:
    f.write(key.public_key().public_bytes(serialization.Encoding.PEM,
                                          serialization.PublicFormat.SubjectPublicKeyInfo))

print("=" * 70)
print("RSA-4096 key pair generated")
print("=" * 70)
print(f"  public : {pub_path}   (commit this)")
print(f"  private: {priv_path}  (DO NOT COMMIT - already in .gitignore)")
print("\nNext steps:")
print("  1. GitHub repo > Settings > Secrets and variables > Actions > New secret")
print("     Name: PRIVATE_KEY   Value: full content of private_key.pem (incl. BEGIN/END lines)")
print("  2. python scripts/encrypt_ground_truth.py")
print("  3. Back up private_key.pem offline and delete it from shared machines.")
