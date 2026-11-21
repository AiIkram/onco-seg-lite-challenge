#!/usr/bin/env python3
"""
ONCO-SEG Lite - one-shot organiser setup. Works on Windows, macOS and Linux.
Run ONCE, locally, before pushing to GitHub.

Windows (PowerShell or cmd):
    python setup_competition.py "D:\\competition gnn cb img\\Dataset_BUSI_with_GT"
macOS / Linux:
    python3 setup_competition.py ~/Downloads/Dataset_BUSI_with_GT

Steps: requirements -> RSA keys -> split -> encrypted test masks -> baseline (CPU, seed 25)
       -> baseline scored on the hidden test set -> leaderboard -> safety check.

Options:
    --epochs 40          baseline training epochs
    --test-shift         build the Hard variant (acquisition shift on the test set)
    --secret STRING      reuse a split secret (default: random, saved to data/private/SPLIT_SECRET.txt)
    --skip-install       don't pip-install requirements
"""
import argparse
import os
import secrets
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")  # Windows: no cp1252 surprises


def step(n, msg):
    print(f"\n==> {n}/8 {msg}", flush=True)


def run(args, cwd=ROOT):
    r = subprocess.run(args, cwd=cwd, env=ENV)
    if r.returncode != 0:
        sys.exit(f"\nFAILED: {' '.join(map(str, args))}\nFix the error above and re-run "
                 "(the script is safe to re-run; keys and the split secret are reused).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("busi_dir", help="path to Dataset_BUSI_with_GT (contains benign/, malignant/, normal/)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--test-shift", action="store_true")
    ap.add_argument("--secret", default=None)
    ap.add_argument("--skip-install", action="store_true")
    a = ap.parse_args()

    raw = os.path.abspath(os.path.expanduser(a.busi_dir.strip().strip('"').strip("'")))
    missing = [c for c in ("benign", "malignant", "normal") if not os.path.isdir(os.path.join(raw, c))]
    if missing:
        # common case: user pointed at the parent folder of Dataset_BUSI_with_GT
        nested = os.path.join(raw, "Dataset_BUSI_with_GT")
        if os.path.isdir(nested) and all(os.path.isdir(os.path.join(nested, c))
                                         for c in ("benign", "malignant", "normal")):
            raw = nested
        else:
            sys.exit(f"'{raw}' must contain benign/, malignant/ and normal/ (missing: {missing}).")
    print(f"BUSI folder: {raw}")

    step(1, "Installing requirements")
    if a.skip_install:
        print("skipped")
    else:
        run([PY, "-m", "pip", "install", "-q", "-r", os.path.join("starter_code", "requirements.txt")])

    step(2, "Generating RSA-4096 key pair")
    if os.path.exists(os.path.join(ROOT, "encryption", "private_key.pem")):
        print("encryption/private_key.pem already exists - keeping it.")
    else:
        run([PY, os.path.join("encryption", "generate_keys.py")])

    step(3, "Building the split (dedup, anti-contamination modification, opaque IDs)")
    priv = os.path.join(ROOT, "data", "private")
    os.makedirs(priv, exist_ok=True)
    secret_file = os.path.join(priv, "SPLIT_SECRET.txt")
    if a.secret:
        secret = a.secret
    elif os.path.exists(secret_file):
        secret = open(secret_file, encoding="utf-8").read().strip()
    else:
        secret = secrets.token_urlsafe(24)
    with open(secret_file, "w", encoding="utf-8") as f:
        f.write(secret)
    cmd = [PY, os.path.join("scripts", "prepare_data.py"), "--raw", raw, "--secret", secret]
    if a.test_shift:
        cmd.append("--test-shift")
    run(cmd)

    step(4, "Encrypting the hidden test masks")
    run([PY, os.path.join("scripts", "encrypt_ground_truth.py")])

    step(5, f"Training the official baseline (CPU, seed 25, {a.epochs} epochs - 20-40 min)")
    sc = os.path.join(ROOT, "starter_code")
    runs_log = os.path.join(sc, "runs_log.csv")
    if os.path.exists(runs_log):
        os.remove(runs_log)
    run([PY, "train_baseline.py", "--run-name", "baseline", "--epochs", str(a.epochs)], cwd=sc)
    run([PY, "generate_submission.py", "--name", "organiser-baseline", "--model-type", "baseline",
         "--ckpt", os.path.join("checkpoints", "baseline.pt")], cwd=sc)

    step(6, "Scoring the baseline on the hidden test set")
    run([PY, os.path.join("scripts", "add_baseline.py")])

    step(7, "Rendering the leaderboard")
    run([PY, os.path.join("competition", "render_leaderboard.py")])
    for p in (os.path.join(ROOT, "submissions", "organiser-baseline.csv"), runs_log):
        if os.path.exists(p):
            os.remove(p)

    step(8, "Safety check")
    if shutil.which("git"):
        if subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT,
                          capture_output=True).returncode != 0:
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=ROOT)
        bad = [p for p in ("encryption/private_key.pem", "data/private/test_ground_truth.npz",
                           "data/private/SPLIT_SECRET.txt")
               if subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT).returncode != 0]
        if bad:
            sys.exit(f"DANGER: not git-ignored: {bad}. Fix .gitignore before committing.")
        print("Private key, split secret and test masks are git-ignored. OK.")
    else:
        print("git not found - install Git for Windows (https://git-scm.com) before pushing.")

    print("""
=============================================================================
 Setup complete.

 KEEP OFFLINE (never commit, back up somewhere safe):
   encryption/private_key.pem       -> GitHub secret PRIVATE_KEY
   data/private/SPLIT_SECRET.txt    -> reproduces the exact split
   data/private/ (test masks, ID mapping, build_info.json)

 Next: GITHUB_SETUP.md, step 3 (create repo, push, add secret, enable Pages).
=============================================================================""")


if __name__ == "__main__":
    main()
