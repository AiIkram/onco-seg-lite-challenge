# Putting ONCO-SEG Lite on GitHub (organiser, ~20 minutes)

Target: **https://github.com/AiIkram/onco-seg-lite-challenge**
Website: **https://aiikram.github.io/onco-seg-lite-challenge/**

## 1. Download BUSI
Download the Kaggle mirror `aryashah2k/breast-ultrasound-images-dataset` or the authors' release, and unzip it **outside** this folder. You need the folder `Dataset_BUSI_with_GT/` with `benign/`, `malignant/` and `normal/` inside.

## 2. Run the one-shot setup (locally)
**Windows (PowerShell, from the repository folder):**
```powershell
cd "D:\competition gnn cb img\onco-seg-lite-challenge"
python setup_competition.py "D:\competition gnn cb img\Dataset_BUSI_with_GT"
```
**macOS / Linux:** `python3 setup_competition.py ~/Downloads/Dataset_BUSI_with_GT`
(or `./setup_competition.sh <path>`; the `.sh` file does not run in PowerShell)

It takes 20–40 minutes on a laptop CPU, mostly baseline training. It creates the keys, the de-duplicated and modified split, the encrypted test masks and the baseline leaderboard row, then checks that nothing secret will be committed. It is safe to re-run: the keys and split secret are reused. If `python` is not found on Windows, install Python 3.11 or 3.12 from python.org and tick **"Add python.exe to PATH"**.

**Back up these three things offline** (password manager or private drive). They are git-ignored:
`encryption/private_key.pem`, `data/private/SPLIT_SECRET.txt`, `data/private/`

## 3. Create the repository and push
On github.com, choose **New repository**, name it `onco-seg-lite-challenge`, make it **Public**, and do **not** add a README or license. Then:
```bash
git add -A
git status            # must NOT list private_key.pem or anything under data/private/
git commit -m "ONCO-SEG Lite challenge: initial release"
git branch -M main
git remote add origin https://github.com/AiIkram/onco-seg-lite-challenge.git
git push -u origin main
```
(With the GitHub CLI: `gh repo create AiIkram/onco-seg-lite-challenge --public --source . --push`.)

## 4. Repository settings
1. **Secret:** Settings → Secrets and variables → Actions → *New repository secret*. Name `PRIVATE_KEY`; value = the **entire** content of `encryption/private_key.pem`, including the `-----BEGIN/END PRIVATE KEY-----` lines.
   CLI: `gh secret set PRIVATE_KEY < encryption/private_key.pem`
2. **Actions permissions:** Settings → Actions → General.
   - *Workflow permissions* → **Read and write permissions**.
   - *Fork pull request workflows from outside collaborators* → **Require approval for first-time contributors who are new to GitHub**. This is enough: the scoring workflow never runs PR code.
3. **GitHub Pages:** Settings → Pages → Source *Deploy from a branch* → `main` / `/docs` → Save. After ~1 minute, check https://aiikram.github.io/onco-seg-lite-challenge/
4. **Discussions** (for participants' questions): Settings → General → Features → ☑ Discussions.
5. **Private vulnerability reporting** (confidential reports, as referenced in RULES.md): Settings → Code security → ☑ Private vulnerability reporting.
6. **Optional branch protection:** Settings → Rules → New branch ruleset for `main`. Require PRs and code-owner review, and add **github-actions** to the *bypass list*. Without that bypass the bot cannot push leaderboard updates.

## 5. Dry-run with a second account (5 minutes, strongly recommended)
From another GitHub account (or ask a colleague):
```bash
# fork AiIkram/onco-seg-lite-challenge, clone the fork, then:
pip install -r starter_code/requirements.txt
cp data/public/sample_submission.csv submissions/dryrun_test.csv
python encryption/encrypt_submission.py submissions/dryrun_test.csv
git add submissions/encrypted/dryrun_test.enc && git commit -m "dry run" && git push
# open a PR to AiIkram/onco-seg-lite-challenge
```
Within about 2 minutes the bot should comment with a score (≈ the fraction of normal test images, since every mask is empty), update the leaderboard and close the PR. Then delete the test row from `docs/leaderboard.csv` on `main`; the *Update Leaderboard* workflow re-renders automatically.

## 6. Announce
Share the website link. The rules (deadline **31 January 2027, 23:59 AoE**, one submission per account) are already in RULES.md, and the bot enforces both the limit and the deadline (`MAX_SUBMISSIONS`, `DEADLINE_UTC` in `.github/workflows/score_submission.yml`).

## 7. LLM arm
Follow `llm_eval/README.md`: one greedy non-agent run per model, plus agent runs per `llm_eval/AGENT_PROTOCOL.md`. Submit valid runs from a dedicated account, e.g. `llm-arm-<model>`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Workflow red: "PRIVATE_KEY secret missing" | Set the secret (step 4.1) and re-run the job |
| Bot can't push (`protected branch`) | Add github-actions to the ruleset bypass list (step 4.6) |
| PR comment says "Could not decrypt" | The participant used an old `public_key.pem`; they must pull from `main` |
| Website 404 | Pages source must be `main` + `/docs`; wait 1–2 minutes |
| Leaderboard page stale | It reads the raw CSV from `main`; hard-refresh the page |
| Need to remove an entry | Delete its row in `docs/leaderboard.csv` on `main` and announce it in Discussions |
