"""
Hybrid encryption used for submissions AND for the hidden test masks.

    payload = gzip(plaintext)
    AES-256-GCM(payload) with a random key + 12-byte nonce   (confidentiality + integrity)
    RSA-4096-OAEP(SHA-256)(aes_key)                          (only the organiser can unwrap)

File layout:  b"OSL1" | len(enc_key) (4 bytes, big-endian) | enc_key | nonce (12) | ciphertext+tag
"""
import gzip
import os

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"OSL1"
_OAEP = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                     algorithm=hashes.SHA256(), label=None)


def load_public_key(pem_bytes: bytes):
    return serialization.load_pem_public_key(pem_bytes)


def load_private_key(pem_bytes: bytes):
    return serialization.load_pem_private_key(pem_bytes, password=None)


def encrypt_bytes(plaintext: bytes, public_key) -> bytes:
    aes_key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(nonce, gzip.compress(plaintext, mtime=0), MAGIC)
    enc_key = public_key.encrypt(aes_key, _OAEP)
    return MAGIC + len(enc_key).to_bytes(4, "big") + enc_key + nonce + ciphertext


def decrypt_bytes(blob: bytes, private_key) -> bytes:
    if blob[:4] != MAGIC:
        raise ValueError("Not an ONCO-SEG Lite encrypted file (bad header). "
                         "Use encryption/encrypt_submission.py.")
    klen = int.from_bytes(blob[4:8], "big")
    enc_key = blob[8:8 + klen]
    nonce = blob[8 + klen:8 + klen + 12]
    ciphertext = blob[8 + klen + 12:]
    aes_key = private_key.decrypt(enc_key, _OAEP)
    return gzip.decompress(AESGCM(aes_key).decrypt(nonce, ciphertext, MAGIC))
