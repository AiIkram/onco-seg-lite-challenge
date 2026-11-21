# 📜 Competition Rules & Guidelines

## 🎯 Objective

Build a **lightweight** segmentation model that delineates breast lesions in ultrasound images and achieves the highest **mean per-image Dice** on the hidden test set.

---

## ✅ Requirements

### 1. Development protocol (identical to GNN-CB, §3.2)

| Constraint | Limit |
|---|---|
| Compute | **3 CPU-hours** of training in total across all your runs. **No GPU.** |
| Tuning runs | **At most 5** training runs (hyperparameter configurations). Pick the final model using **validation data only**. |
| Seed | **25** for every run |
| Convergence | Plot the training-loss curve and check it. Early stopping is allowed. |
| Documentation log | Required: search strategy, every configuration tried, runtime, hardware, loss-curve check and any LLM-generated snippet. Use `submissions/REPORT_TEMPLATE.md`; it goes in your PR as `submissions/reports/<name>.md`. |
| Final submission | **One**, evaluated once on the hidden test set. The PR is closed automatically after scoring. |
| Inference | ≤ 5 minutes for the full test set on a laptop CPU |
| Model size | **≤ 5,000,000 parameters**, counting **every** network used at inference (ensemble members, encoders, heads) |
| No copy-paste | Do not copy external solutions or repositories for this task |
| Designers | The competition designer does not take part in their own competition |

`starter_code/train_baseline.py` enforces most of this for you. It uses seed 25, runs on CPU by default, and stops at a time budget. It saves the loss curve and appends every run to `runs_log.csv`, which `generate_submission.py` turns into a draft documentation log.

**Allowed architectures:** any CNN or transformer (U-Net, U-Net++, attention U-Net, DeepLab-lite, SegFormer-B0-scale, …) that fits the budget.

**Pretrained weights: none.** Every model is trained **from scratch** on the provided data within the 3 CPU-hour budget. This keeps humans and LLMs on equal terms, because the LLM sandbox has no internet. It excludes ImageNet encoders and medical or segmentation foundation models (SAM, MedSAM, SAM-Med2D, BiomedParse, …), including models distilled from them.

### 2. Data usage

**Allowed**
- ✅ All provided training images, masks and diagnosis labels
- ✅ Any validation scheme or cross-validation on the training set
- ✅ Any augmentation computed from the training set
- ✅ Synthetic images, if generated only from the training set
- ✅ Self-supervised or test-time techniques that use **test images only**, never test labels (e.g. test-time augmentation)

**Prohibited**
- ❌ External breast-ultrasound datasets (BUSI originals, BUS-UCLM, UDIAT/Dataset B, BUSI_WHU, …)
- ❌ **Any attempt to identify test images or retrieve their masks.** BUSI is public, so matching test images to the original files is cheating and leads to disqualification.
- ❌ Manual annotation, or manual correction of predictions on test images
- ❌ Probing the scorer, for example with multiple accounts

### 3. Reproducibility
- Seed 25 everywhere. Record hyperparameters in the documentation log.
- Keep your training and inference code. **Top-3 finishers must publish it** (repository link or PR) together with a short README: architecture, parameter count, training time, hardware and the LLM used if any. Organisers may re-run it.

---

## 📤 Submission requirements

### CSV format
- Header: `image_id,rle`
- Exactly **one row per ID** in `data/public/test_ids.csv`, with no duplicates and no extra IDs
- `rle`: row-major, 1-indexed `start length` pairs, sorted and non-overlapping, inside the 256×256 image. **An empty string means an empty mask.**

### What goes in the Pull Request
- `submissions/encrypted/<name>.enc`: required, new file, created with `encryption/encrypt_submission.py`
- `submissions/metadata/<name>.json`: recommended. `generate_submission.py` writes it with `model_type`, `architecture`, `params`, `llm`, `seed`, `n_tuning_runs`, `train_minutes_total` and `hardware`.
- `submissions/reports/<name>.md`: your documentation log (required by the protocol)
- **Nothing else.** A PR that touches any other file is rejected without being scored.

`<name>` must be 2–60 characters: letters, digits, `_` and `-`.

---

## 🏆 Evaluation

**Primary metric: mean per-image Dice**

```
Dice = 2|P∩G| / (|P|+|G|);   both empty → 1;   exactly one empty → 0
Score = mean over all test images
```

Secondary metrics (IoU, lesion-only Dice, specificity on normal scans, sensitivity, per-class Dice) are shown for analysis but not used for ranking.

### Ranking
- Each participant is ranked by their **best** scored submission.
- Equal scores share a rank; the earlier submission is listed first.
- Scores are given to 4 decimals.

---

## 🔄 Submission policy

- **Scored submissions per GitHub account: 1** (GNN-CB: one final submission per competition version). The organisers set this with `MAX_SUBMISSIONS` in `.github/workflows/score_submission.yml` and announce it before launch.
- Attempts that fail before scoring do **not** count: invalid format, decryption error or a PR with disallowed files.
- One account per person. Teams of up to 3 submit from **one** account and list their members in the PR description.

---

## 🤖 Human vs. LLM track

Declare honestly in your metadata:

| `model_type` | Meaning |
|---|---|
| `human` | Code and design decisions by people. An LLM may help **only** with low-level utility functions that contain no task-specific information (e.g. "write an RLE encoder", "plot a loss curve"). Every LLM-generated snippet must be listed in the documentation log. |
| `llm-only` | An LLM or agent wrote the solution end to end. A human only ran commands or copied files, without changing code or choices. |
| `human+llm` | Substantial use of an LLM together with human design or debugging |

LLM entries should name the model in `llm` (e.g. `"Claude Opus 4.7"`) and keep the transcript. The organisers' LLM arm runs under the frozen GNN-CB protocol in `llm_eval/`: zero-shot frozen prompt, plan-then-code, CPU only, at most 5 repairs, one greedy run. LLM entries follow **exactly the same rules** (data, parameter limit, one submission).

---

## 🚫 Violations

**Disqualification:** test-image identification or mask retrieval · external breast-ultrasound data · foundation models · GPU training or undeclared budget overruns · task-level LLM use in the `human` track · multiple accounts

**Warning, then disqualification:** missing or broken code for a top-3 entry · wrong `model_type` · parameter count misreported by more than 5%

Organisers may audit any entry and remove it from `docs/leaderboard.csv`. Removals are announced publicly.

---

## 🤝 Collaboration

**Encouraged:** discussing ideas, papers and architectures in Discussions · sharing general tips · asking questions

**Not allowed between participants or teams:** sharing code, weights, predictions or validation scores tuned to the test set

---

## 📝 Submission checklist

- [ ] `python starter_code/check_submission.py submissions/<name>.csv` prints **VALID**
- [ ] Parameters ≤ 5M, counted with `sum(p.numel() for p in model.parameters())` over every model used
- [ ] Seed 25 · ≤ 5 tuning runs · ≤ 3 CPU-hours of training in total · CPU inference ≤ 5 min
- [ ] Loss curve checked; final model chosen on validation only
- [ ] Documentation log `submissions/reports/<name>.md` completed (incl. LLM-utility usage or "none")
- [ ] No external breast-ultrasound data, no foundation models, no test-label leakage
- [ ] Seeds fixed and code kept
- [ ] `model_type` declared honestly
- [ ] The PR contains only the `.enc` file, the `.json` and the report `.md`

---

## 🎯 Timeline

- **Launch:** 1 December 2026
- **Submission deadline:** 31 January 2027, 23:59 AoE
- **Code release for top 3:** by 7 February 2027

The scoring bot stops accepting submissions automatically at the deadline.
- **Final results announced:** 15 February 2027

## 📧 Questions

GitHub **Discussions** for general questions · **Issues** for technical problems with the pipeline · for anything confidential (e.g. reporting a violation), use GitHub's private vulnerability reporting ([Security → Report a vulnerability](https://github.com/AiIkram/onco-seg-lite-challenge/security/advisories/new)) or contact [@AiIkram](https://github.com/AiIkram).

## 🔄 Rule updates

Rules may be clarified but will not be changed retroactively. Changes are announced in Discussions before they apply.

---

**Version:** 1.0
