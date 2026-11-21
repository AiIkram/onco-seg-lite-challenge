#!/usr/bin/env bash
# ============================================================================
#  ONCO-SEG Lite - one-shot organiser setup (run ONCE, locally, before pushing)
#
#    ./setup_competition.sh /path/to/Dataset_BUSI_with_GT
#
#  Does everything that needs the raw data or the private key:
#    1. installs requirements          5. trains the official baseline (CPU, seed 25)
#    2. generates the RSA key pair     6. scores it and adds it to the leaderboard
#    3. builds the public/private split 7. renders leaderboard.md / leaderboard.json
#    4. encrypts the hidden test masks 8. checks nothing secret is about to be committed
#
#  Options (environment variables):
#    SPLIT_SECRET=...     reuse a secret (default: random, saved to data/private/SPLIT_SECRET.txt)
#    TEST_SHIFT=1         build the Hard variant (acquisition shift on the test set)
#    BASELINE_EPOCHS=40   baseline training epochs
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

RAW="${1:-}"
if [ -z "$RAW" ] || [ ! -d "$RAW/benign" ] || [ ! -d "$RAW/malignant" ] || [ ! -d "$RAW/normal" ]; then
  echo "Usage: ./setup_competition.sh /path/to/Dataset_BUSI_with_GT"
  echo "       (the folder must contain benign/, malignant/ and normal/)"
  exit 1
fi
PY="${PYTHON:-python3}"
EPOCHS="${BASELINE_EPOCHS:-40}"
SHIFT_FLAG=""
[ "${TEST_SHIFT:-0}" = "1" ] && SHIFT_FLAG="--test-shift"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }

step "1/8 Installing requirements"
$PY -m pip install -q -r starter_code/requirements.txt 2>/dev/null \
  || $PY -m pip install -q --break-system-packages -r starter_code/requirements.txt

step "2/8 Generating RSA-4096 key pair"
if [ -f encryption/private_key.pem ]; then
  echo "encryption/private_key.pem already exists - keeping it."
else
  $PY encryption/generate_keys.py >/dev/null
  echo "Created encryption/public_key.pem (commit) and encryption/private_key.pem (SECRET)."
fi

step "3/8 Building the split from $RAW"
mkdir -p data/private
if [ -n "${SPLIT_SECRET:-}" ]; then
  SECRET="$SPLIT_SECRET"
elif [ -f data/private/SPLIT_SECRET.txt ]; then
  SECRET="$(cat data/private/SPLIT_SECRET.txt)"
else
  SECRET="$($PY -c 'import secrets; print(secrets.token_urlsafe(24))')"
fi
printf "%s" "$SECRET" > data/private/SPLIT_SECRET.txt
$PY scripts/prepare_data.py --raw "$RAW" --secret "$SECRET" $SHIFT_FLAG

step "4/8 Encrypting the hidden test masks"
$PY scripts/encrypt_ground_truth.py

step "5/8 Training the official baseline (CPU, seed 25, $EPOCHS epochs)"
rm -f starter_code/runs_log.csv
( cd starter_code && $PY train_baseline.py --run-name baseline --epochs "$EPOCHS" )
( cd starter_code && $PY generate_submission.py --name organiser-baseline --model-type baseline \
    --ckpt checkpoints/baseline.pt )

step "6/8 Scoring the baseline on the hidden test set"
$PY scripts/add_baseline.py

step "7/8 Rendering the leaderboard"
$PY competition/render_leaderboard.py
rm -f submissions/organiser-baseline.csv starter_code/runs_log.csv

step "8/8 Safety check"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || git init -q -b main
LEAK=0
for f in encryption/private_key.pem data/private/test_ground_truth.npz data/private/SPLIT_SECRET.txt; do
  if ! git check-ignore -q "$f"; then echo "DANGER: $f is NOT git-ignored!"; LEAK=1; fi
done
if [ $LEAK -ne 0 ]; then echo "Fix .gitignore before committing."; exit 1; fi
echo "Private key, split secret and test masks are git-ignored. OK."

cat <<EOF

=============================================================================
 Setup complete.

 KEEP OFFLINE (never commit, back up somewhere safe):
   encryption/private_key.pem       -> paste into GitHub secret PRIVATE_KEY
   data/private/SPLIT_SECRET.txt    -> reproduces the exact split
   data/private/test_ground_truth.npz, test_mapping.csv, build_info.json

 Next: follow GITHUB_SETUP.md (create repo, push, add secret, enable Pages).
=============================================================================
EOF
