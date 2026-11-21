# ONCO-SEG Lite Competition Template – Designer Manual

This manual explains how to set up and run the ONCO-SEG Lite image-segmentation competition on GitHub, with encrypted submissions, automatic scoring and a live leaderboard. It follows the structure of the *GNN Coding Competition Template – Designer Manual* and notes where the image version differs.

---

## 1. Purpose of the template

The template evaluates modelling capability on a **low-compute medical image segmentation** task without executing participant code. Participants submit **encrypted prediction files only**. GitHub Actions decrypts them in memory, scores them against hidden test masks and updates the leaderboard. The same pipeline supports **human vs. LLM** studies: every entry declares `human`, `llm-only` or `human+llm`.

## 1b. Alignment with the GNN-CB protocol

This competition is designed as an **image-domain counterpart of GNN-CB** (Human vs. LLM competition benchmark) and reproduces its protocol:

| GNN-CB | ONCO-SEG Lite |
|---|---|
| One repository per competition, hidden test set, automated scoring | same |
| Public-key encryption of predictions; decryption only in CI | same (RSA-4096 + AES-GCM, in-memory) |
| One final submission, evaluated once; malformed or undecryptable files rejected | `MAX_SUBMISSIONS=1`; rejected attempts don't count; PR closed after scoring |
| Kaggle-style ties | same |
| Anti-contamination: no canonical configuration (noise injection, permutation, site shift) | re-crop + gamma + speckle on all images; optional test-only acquisition shift (`--test-shift`) |
| Human protocol: seed 25, 3 CPU-hours, ≤ 5 tuning runs, validation-only selection, loss curves, early stopping | same, built into `starter_code/train_baseline.py` |
| Domain restriction ("GNN only") | segmentation network trained from scratch, ≤ 5M parameters, no foundation models |
| LLM use in the human track only for low-level utilities, logged | same (`submissions/REPORT_TEMPLATE.md`) |
| Documentation log (search strategy, configs, runtime, hardware, loss curves, LLM use) | `submissions/reports/<name>.md`, accepted in the PR |
| Designer excluded from own competition | same |
| LLM arm: frozen prompt with slots, plan-then-code, temperature 0, 10K tokens, CPU-only, repair k ≤ 5, one greedy run; agent and non-agent | `llm_eval/` (`run_nonagent.py`, `AGENT_PROTOCOL.md`) |
| Reported: score, gap to Human Top, human avg/std/high/low | leaderboard shows Δ Human Top; `docs/leaderboard.csv` has everything needed for the tables |

The specification for the paper's appendix is in `COMPETITION_CARD.md`.

## 2. Contents

| Path | Role | Who |
|---|---|---|
| `data/public/` | train images and masks, test images, `test_ids.csv`, sample submission | committed |
| `data/private/` | plaintext test masks + ID mapping | **never committed** (git-ignored) |
| `data/private_encrypted/test_ground_truth.enc` | test masks encrypted with the public key | committed |
| `competition/metrics.py` | **official** Dice/IoU, RLE codec, validation | frozen after launch |
| `competition/evaluate.py` | offline scoring with the plaintext ground truth | organisers |
| `competition/render_leaderboard.py` | CSV → `leaderboard.md` + `docs/leaderboard.json` | CI |
| `encryption/` | key generation, participant encryption, organiser decryption | mixed |
| `scripts/prepare_data.py` | raw BUSI → public/private split | organisers |
| `scripts/encrypt_ground_truth.py` | encrypt the test masks | organisers |
| `scripts/ci_inspect_pr.py`, `scripts/ci_score.py` | PR checks + scoring | CI |
| `scripts/make_dummy_busi.py` | fake dataset for dry runs | organisers |
| `starter_code/` | TinyUNet baseline (CPU, seed 25, run log, loss curves), submission generator, format checker | participants |
| `submissions/REPORT_TEMPLATE.md` | documentation log template (protocol) | participants |
| `llm_eval/` | frozen prompt template, prompt builder, non-agent harness, agent protocol, mock responses | organisers |
| `COMPETITION_CARD.md` | Appendix-A-style specification for the paper | organisers |
| `docs/` | GitHub Pages site: `index.html`, `leaderboard.html/.css/.js`, **`leaderboard.csv`** | CI + organisers |
| `.github/workflows/score_submission.yml` | scores PRs, posts comment, updates leaderboard | CI |
| `.github/workflows/update_leaderboard.yml` | re-renders after manual CSV edits | CI |

## 3. One-time setup (≈ 30 minutes)

> **Shortcut:** `python setup_competition.py /path/to/Dataset_BUSI_with_GT` (Windows, macOS, Linux) runs steps 3.2–3.4 in one go: keys, split, encrypted masks, baseline and leaderboard. Then follow **`GITHUB_SETUP.md`** to push and configure the repository. The sections below explain what the script does.

### 3.1 Get the data
Download **BUSI** (`Dataset_BUSI_with_GT`, 780 images: 437 benign, 210 malignant, 133 normal) from its official source and check its terms of use. Keep it **outside** the repository, or under `data/raw/`, which is git-ignored.

### 3.2 Generate keys
```bash
pip install -r starter_code/requirements.txt
python encryption/generate_keys.py
```
- `encryption/public_key.pem` → **commit it**
- `encryption/private_key.pem` → copy its full content (including the BEGIN/END lines) into **Settings → Secrets and variables → Actions → New repository secret**, name `PRIVATE_KEY`. Then back it up offline. It is git-ignored.

### 3.3 Build the split
```bash
python scripts/prepare_data.py --raw /path/to/Dataset_BUSI_with_GT --secret "<long random string>"
python scripts/encrypt_ground_truth.py
```
Store the `--secret` in a password manager. The same secret and the same raw data reproduce the exact split and IDs.

`prepare_data.py` merges multi-lesion masks, groups near-duplicate images (BUSI is known to contain duplicates) so that they never straddle train and test, and stratifies by class at about 80/20. It then applies the **anti-contamination modification** (a lesion-preserving re-crop plus gamma and speckle noise, seeded per image from your secret), resizes to 256×256 and renames every file to `img_<HMAC>`. Add `--test-shift` for the Hard variant. The settings used are saved to `data/private/build_info.json`; report them in the paper.

### 3.4 Post the official baseline
```bash
cd starter_code && python train_baseline.py && python generate_submission.py --name organiser-baseline --model-type human && cd ..
python competition/evaluate.py --submission submissions/organiser-baseline.csv
```
Then either submit it through a PR like a participant (recommended, because it tests the whole pipeline) or append it to `docs/leaderboard.csv` by hand.

### 3.5 Push and enable services
1. Commit everything except what `.gitignore` excludes. Verify with `git status` that no `data/private/` file or `.pem` other than `public_key.pem` is staged.
2. **Settings → Actions → General:** allow Actions; under *Workflow permissions* choose **Read and write**.
3. **Settings → Pages:** Source = *Deploy from branch*, branch `main`, folder `/docs`.
4. If `main` is protected, allow `github-actions[bot]` to push (ruleset bypass), or the leaderboard commit will fail.
5. Pull requests from forks: *Settings → Actions → General → Fork pull request workflows*. The scoring workflow uses `pull_request_target`, so it runs with secrets even for first-time contributors.

### 3.6 Dry run (strongly recommended)
Before using real data, rehearse with fake data on a throw-away repository:
```bash
python scripts/make_dummy_busi.py --out /tmp/Dummy_BUSI --n 300
python scripts/prepare_data.py --raw /tmp/Dummy_BUSI --secret "dry-run-secret-123"
python scripts/encrypt_ground_truth.py
```
Then open a test PR from a second account.

## 4. Preparing the data: what is public, what is hidden

| Public (committed) | Hidden |
|---|---|
| train images, masks, diagnosis; test images; test IDs | test masks (plaintext only on your machine, encrypted in the repo) |

**Why not a GitHub secret?** The GNN template stored labels in a `TEST_LABELS` secret. Secrets are limited to 48 KB, and segmentation masks are larger, so the masks are encrypted with the challenge public key and committed. Only the `PRIVATE_KEY` secret inside CI can open them. If that key leaks, rotate it (section 10).

## 5. Submission format

`image_id,rle`, one row per test ID; row-major, 1-indexed `start length` runs; empty string = no lesion. See `competition/metrics.py`.

## 6. How participants submit

1. Write `submissions/<name>.csv` and check it with `starter_code/check_submission.py`.
2. `python encryption/encrypt_submission.py submissions/<name>.csv` validates the format and then writes `submissions/encrypted/<name>.enc`.
3. Optionally add `submissions/metadata/<name>.json` with `model_type`, `architecture`, `params` and `llm`.
4. Open a PR containing **only** those files.

## 7. Automatic scoring (`score_submission.yml`)

| Step | What happens |
|---|---|
| Checkout | the **default branch** only. PR code is never checked out or executed. |
| Inspect | the PR files API must show exactly one *new* `submissions/encrypted/<name>.enc` and optionally `submissions/metadata/<name>.json`, and nothing else |
| Extract | `git show pr-head:<path>` copies the two blobs to `$RUNNER_TEMP`. Files over 5 MB are rejected. |
| Score | `scripts/ci_score.py` checks the submission limit, decrypts the submission and ground truth **in memory**, validates, scores and appends to `docs/leaderboard.csv` |
| Publish | renders `leaderboard/leaderboard.md` and `docs/leaderboard.json`, commits and pushes to `main` (retried with rebase) |
| Comment | posts the score table, rank and submissions used, or a precise reason for rejection |

A `concurrency` group serialises scoring jobs so that two PRs never overwrite each other's leaderboard rows. Rejected attempts (bad files, decryption or format errors) do **not** count towards the limit.

**Submission limit:** `MAX_SUBMISSIONS` at the top of the workflow (default `1`, as in the GNN template). The leaderboard ranks each participant by their best entry.

## 8. Leaderboard

- **Source of truth:** `docs/leaderboard.csv` with columns `team,score,iou,lesion_dice,specificity,model_type,architecture,params,llm,date,run_id`.
- **Static view:** `leaderboard/leaderboard.md`, auto-generated. Do not edit it.
- **Interactive view:** `docs/leaderboard.html` has search, model-type filter, date filter, a best-per-participant/all-submissions toggle, sortable columns, column toggles, and stat tiles (best overall, best human, best LLM-only). On `*.github.io` it reads the CSV from `raw.githubusercontent.com`, so new scores appear without waiting for a Pages rebuild.

To **remove or correct** an entry (e.g. disqualification), edit `docs/leaderboard.csv` on `main`. `update_leaderboard.yml` re-renders automatically. You can also run it by hand from the Actions tab.

## 9. Using the template for research (human vs. LLM)

- Give every arm the same budget: time, the 5M-parameter cap, the compute and the number of submissions.
- Run LLM arms from separate GitHub accounts or teams with `model_type: llm-only` and `llm: <model name>`. Keep the transcripts.
- Report: validity rate (rejected/total attempts, from the PR comments), best score, the gap between lesion Dice and specificity (whether errors come from over-segmenting or from false positives), Dice per class, and score versus parameter count.
- For offline arms such as batch-evaluated LLM runs, score them with `competition/evaluate.py` against the plaintext masks and append the rows to the CSV by hand.

## 9b. Running the LLM arm

See `llm_eval/README.md`. In short: `python llm_eval/build_prompt.py`, read the prompt once and freeze it, then run one greedy `run_nonagent.py` per model with temperature 0, 10K tokens and k ≤ 5. Agent runs follow `llm_eval/AGENT_PROTOCOL.md`. Valid runs produce a ready-to-submit CSV, metadata and report. Invalid runs are recorded in `llm_eval/results.csv` for the validity rate. If you have the exact GNN-CB Listing 1 text, paste it into `llm_eval/prompt_template.md`, change "GNN" to "image segmentation" and keep the slot names, so that the two benchmarks share a byte-identical frozen prompt.

## 10. Common pitfalls

| Pitfall | Prevention |
|---|---|
| Committing `data/private/` or `private_key.pem` | `.gitignore` covers them. Check `git status` before every push. If the key leaks, run `generate_keys.py --force`, update the secret, re-run `encrypt_ground_truth.py` and ask everyone to pull the new public key. |
| **Public-source leakage** | BUSI is public, so a determined cheater can match test images to the originals. Opaque IDs, re-cropping and noise injection make this harder and remove pixel identity, but do not stop it. Rely on the rules, require code from the top 3, and re-run suspicious entries. For high stakes, use a private test set. |
| Changing `metrics.py` after launch | Freeze it. Protect `competition/` and `scripts/` with CODEOWNERS and branch rules. |
| Bot cannot push | Enable read/write workflow permissions and add a bypass for branch protection. |
| PR scored twice after `synchronize` | Each scoring counts as a submission, and the limit check stops extra ones. Tell participants to open the PR only when the file is final. |
| Changing the split mid-competition | Never. The leaderboard would no longer be comparable. |

## 11. Customisation

| Change | Where |
|---|---|
| Metric (e.g. add HD95, change the empty-mask convention) | `competition/metrics.py` → `evaluate_masks()` (`score` is the ranked key) |
| Image size | `IMAGE_SIZE` in `metrics.py`, then re-run `prepare_data.py` |
| Another 2D oncology dataset (e.g. Kvasir-SEG polyps, ISIC skin lesions, 2D slices of brain MRI) | adapt `collect_cases()` / `load_case()` in `prepare_data.py`. Everything downstream stays the same. |
| Multi-class segmentation | store label maps instead of binary masks, use one RLE column per class, and compute mean Dice over classes |
| Submission limit | `MAX_SUBMISSIONS` in the workflow |
| Parameter cap | `PARAM_LIMIT` in `scripts/ci_score.py` (declared, flagged), README and RULES |
| Look of the leaderboard | `docs/leaderboard.css` / `.js` (column list in `COLUMNS`) |

Document every change and avoid changing rules after the first submission.

## 12. Verification checklist before launch

- [ ] `encryption/public_key.pem` committed; `PRIVATE_KEY` secret set
- [ ] `data/private_encrypted/test_ground_truth.enc` committed; `data/private/` **not** committed
- [ ] `data/public/train/`, `test/`, `test_ids.csv` and `sample_submission.csv` present
- [ ] Actions enabled with read/write permissions; Pages serving `main` `/docs`
- [ ] `docs/leaderboard.html`, `docs/leaderboard.css` and `docs/leaderboard.js` exist
- [ ] A test PR from a second account was scored and the comment posted
- [ ] The baseline entry is on the leaderboard
- [ ] Timeline and `MAX_SUBMISSIONS` announced in RULES.md

## 13. Architecture summary

```
participant ──CSV──▶ encrypt_submission.py ──.enc──▶ Pull Request
                                                     │ pull_request_target (default-branch code only)
                                                     ▼
                         ci_inspect_pr.py ─▶ ci_score.py ──(in-memory decrypt: submission + GT)
                                                     │
                       docs/leaderboard.csv ◀────────┘  (single source of truth)
                                │
             ┌──────────────────┼────────────────────┐
             ▼                  ▼                    ▼
   leaderboard/leaderboard.md  docs/leaderboard.json  docs/leaderboard.html (Pages, live)
```
