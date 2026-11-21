# 🎗️ ONCO-SEG Lite Challenge: Breast Tumour Segmentation on a Budget

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
![GPU](https://img.shields.io/badge/GPU-optional-success.svg)
![Params](https://img.shields.io/badge/params-%E2%89%A45M-orange.svg)

**🌐 Website:** [aiikram.github.io/onco-seg-lite-challenge](https://aiikram.github.io/onco-seg-lite-challenge/) · **🏆 Leaderboard:** [live](https://aiikram.github.io/onco-seg-lite-challenge/leaderboard.html) · [leaderboard.md](leaderboard/leaderboard.md) · **📜 Rules:** [RULES.md](RULES.md) · **🛠 Organisers:** [DESIGNER_MANUAL.md](DESIGNER_MANUAL.md)

---

## 🎯 Challenge overview

Welcome to **ONCO-SEG Lite**. The task is to **segment breast tumours in ultrasound images** with **lightweight models** (at most 5M parameters), trained **on a CPU in under 3 hours**.

Breast ultrasound is cheap, portable and widely used for breast cancer screening in low-resource settings, but reading the images depends heavily on the operator. A segmentation model that runs on modest hardware is useful in exactly those settings, and that is why this challenge caps compute instead of rewarding the biggest model.

The challenge is the image-segmentation counterpart of the **GNN-CB** competition benchmark (Human vs. LLM evaluation) and uses the same protocol. It also serves as a **human vs. LLM benchmark**. Every submission declares whether it was made by a `human`, by an `llm-only` agent, or by a `human+llm` pair. All three use the same data, the same budget and the same hidden test set.

### 🏆 At a glance

| Aspect | Details |
|---|---|
| **Task** | 2D binary semantic segmentation (lesion vs. background) |
| **Input** | 256×256 grayscale breast ultrasound PNG |
| **Output** | Binary mask per image, run-length encoded in a CSV |
| **Metric** | **Mean per-image Dice** (normal scans: an empty prediction scores 1) |
| **Budget (GNN-CB protocol)** | **CPU only**, 3 CPU-hours of training in total, ≤ 5 tuning runs, seed 25, ≤ 5M parameters |
| **Data** | Derived from BUSI (Al-Dhabyani et al., 2020): benign, malignant and normal scans |
| **Submissions** | Encrypted, scored automatically by GitHub Actions, live leaderboard |

### 🎓 What you will practise

- Encoder–decoder segmentation (U-Net and its lightweight variants)
- Dice/BCE/focal losses and class imbalance at the pixel level
- Avoiding false positives on images with no lesion (post-processing, image-level gating)
- Efficient architectures: depthwise convolutions, MobileNet/EfficientNet-lite encoders, attention gates
- Honest validation on a small medical dataset

---

## 📊 Dataset

| Split | Images | Labels |
|---|---|---|
| `data/public/train/` | ~80% | images + masks + diagnosis (`benign`/`malignant`/`normal`) |
| `data/public/test/` | ~20% | **images only**; masks are encrypted and used only by CI |

- **Images:** `train/images/<id>.png`, 8-bit grayscale, 256×256 (resized from about 500×500 originals). Like GNN-CB, the data is deliberately **non-canonical**: every scan is re-framed and re-textured (gamma and speckle), so it is not pixel-identical to the public release.
- **Masks:** `train/masks/<id>.png`, 0 = background, 255 = lesion. Images with several lesions have them merged into one mask.
- **Metadata:** `train_metadata.csv` (`image_id, diagnosis`). The diagnosis is for training only; you never predict it.
- **Test IDs:** `test_ids.csv`. Use `sample_submission.csv` (all masks empty) as a template.

**Source & citation:** Al-Dhabyani W., Gomaa M., Khaled H., Fahmy A. *Dataset of breast ultrasound images.* Data in Brief 28 (2020) 104863. doi:10.1016/j.dib.2019.104863

---

## 🚀 Getting started

```bash
# 1. Fork https://github.com/AiIkram/onco-seg-lite-challenge on GitHub, then:
git clone https://github.com/$YOUR_GITHUB_USER/onco-seg-lite-challenge.git   # your fork
cd onco-seg-lite-challenge
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r starter_code/requirements.txt

cd starter_code
python train_baseline.py                  # run1: CPU, 128 px, seed 25 (~20-40 min on a laptop)
python train_baseline.py --base 24 --lr 1e-3   # run2 ... at most 5 runs, 3 CPU-hours in total
python generate_submission.py --name your_name # best run -> CSV + metadata + draft documentation log
cd ..
python starter_code/check_submission.py submissions/your_name.csv
```

The baseline is a **TinyUNet** (about 1.9M parameters) trained with BCE + Dice loss, flip and gain augmentation, flip test-time augmentation, and a threshold plus minimum-area filter tuned on validation. Its hidden-test score is the `organiser-baseline` row on the [leaderboard](leaderboard/leaderboard.md). Beat it!

> 💡 **No GPU needed, and none allowed:** like GNN-CB, the protocol is CPU-only. Each run saves `checkpoints/<run>_loss_curve.png` and appends a line to `starter_code/runs_log.csv`, which feeds your documentation log.

---

## 📁 Repository structure

```
├── data/
│   ├── public/                      # train images+masks, test images, ids, sample submission
│   └── private_encrypted/           # test masks, encrypted (only CI can open)
├── starter_code/
│   ├── common.py                    # data loading, TinyUNet, loss, post-processing
│   ├── train_baseline.py            # baseline training
│   ├── generate_submission.py       # test predictions -> submissions/<name>.csv
│   ├── check_submission.py          # local format check
│   └── requirements.txt
├── competition/
│   ├── metrics.py                   # OFFICIAL Dice / IoU / RLE / validation
│   ├── evaluate.py                  # organiser offline scoring
│   └── render_leaderboard.py
├── llm_eval/                        # GNN-CB LLM protocol: frozen prompt, plan-then-code, repair loop
├── encryption/                      # encrypt_submission.py (you), keys & decrypt (organisers)
├── scripts/                         # data preparation + CI scoring (organisers)
├── submissions/
│   ├── encrypted/                   # <name>.enc  ← your predictions (encrypted)
│   ├── metadata/                    # <name>.json ← model type, params, budget
│   ├── reports/                     # <name>.md   ← documentation log (protocol)
│   └── REPORT_TEMPLATE.md
├── leaderboard/leaderboard.md       # auto-generated
├── docs/                            # GitHub Pages site + leaderboard.csv (source of truth)
├── .github/workflows/               # auto-scoring + leaderboard rendering
├── setup_competition.py / .sh      # organiser one-shot setup (Windows: the .py)
├── README.md · RULES.md · DESIGNER_MANUAL.md · GITHUB_SETUP.md · COMPETITION_CARD.md
```

---

## 📤 Making a submission

### Format

A CSV with **one row per test image** and two columns:

```csv
image_id,rle
img_0a1b2c3d4e,32001 14 32257 18 32513 21
img_9f8e7d6c5b,
```

- `rle` is a run-length encoding of the 256×256 mask, flattened **row-major** (pixel `(r, c)` → `r*256 + c`), written as space-separated `start length` pairs with **1-indexed** starts, sorted and non-overlapping.
- **An empty string means "no lesion".** That is the correct answer for normal scans.
- Use `competition.metrics.rle_encode(mask)` so you don't have to implement it yourself.

### Steps

1. **Fork** this repository.
2. Train your model and write `submissions/<your_name>.csv`.
3. **Encrypt** it. The script validates the format first, so a broken file never costs you a submission:
   ```bash
   python encryption/encrypt_submission.py submissions/<your_name>.csv
   ```
4. Complete your **documentation log** `submissions/reports/<your_name>.md` (drafted by `generate_submission.py` from `submissions/REPORT_TEMPLATE.md`). Check the metadata `submissions/metadata/<your_name>.json`:
   ```json
   {
     "model_type": "human",
     "architecture": "TinyUNet-b16",
     "params": 1942289,
     "llm": ""
   }
   ```
   `model_type` is one of `human`, `llm-only` or `human+llm`. Put the LLM's name in `llm` if one was used.
5. Commit **only** `submissions/encrypted/<your_name>.enc`, the JSON and the report, push, and **open a Pull Request**.
6. GitHub Actions decrypts the file in memory, scores it against the hidden masks, posts your score and rank, updates the leaderboard and closes the PR.

> ⚠️ Never commit the plain CSV (`.gitignore` blocks it). A PR that touches any other file is rejected without being scored, and the rejected attempt does not count.

---

## 📈 Evaluation

```
Dice(P, G) = 2|P ∩ G| / (|P| + |G|)        per image
           = 1  if P and G are both empty  (correct "no lesion")
           = 0  if exactly one is empty
Score      = mean Dice over all test images
```

**Why mean per-image Dice?** Every patient counts the same, small lesions matter as much as large ones, and hallucinating a tumour on a healthy scan is penalised.

The bot also reports IoU, Dice on lesion images only, specificity on normal scans, lesion-detection sensitivity, and Dice per class (benign, malignant, normal). Only mean Dice is used for ranking.

---

## 💡 Tips

**Beginner**
- Run the baseline unchanged first, then change one thing at a time.
- Try `--base 24` or `--base 32` (still under 5M) and longer training.
- Hold out a validation split and never tune on the test set; you only get one shot.

**Advanced**
- Efficient from-scratch blocks: depthwise-separable convs, inverted residuals (MobileNet-style), attention gates, deep supervision. No pretrained weights are allowed.
- Loss functions: Dice + focal, Lovász, boundary-aware losses.
- An image-level "lesion present?" head to gate the mask on normal scans.
- Stronger augmentation: elastic deformation, speckle noise, small rotations and scaling.
- k-fold training with a logit-averaged ensemble. All folds together must stay ≤ 5M parameters.
- Multi-scale or test-time augmentation and CRF-free morphological post-processing.

**Common pitfalls**
- ⚠️ Predicting small blobs on normal scans: each one turns that image's score from 1 into 0.
- ⚠️ Overfitting on a few hundred images: use strong augmentation and early stopping on validation Dice.
- ⚠️ RLE mistakes such as column-major order, 0-indexed starts or the wrong image size. Run `check_submission.py`.

---

## 🎯 Rules (summary)

✅ **Must:** CPU only · ≤ 3 CPU-hours of training · ≤ 5 tuning runs · seed 25 · loss curves · documentation log · ≤ 5M parameters · CPU inference ≤ 5 min · declare `model_type` · top 3 share their code

❌ **Cannot:** use a GPU · use pretrained weights (train from scratch) · use an LLM for task-level work in the `human` track (low-level utilities only, logged) · use external breast-ultrasound data or the original BUSI files · try to identify test images or recover their masks · use SAM/MedSAM or other segmentation foundation models · hand-label test images

Full details are in **[RULES.md](RULES.md)**.

---

## 📜 License

Code: MIT ([LICENSE](LICENSE)). Data: BUSI, released for research by Al-Dhabyani et al. (2020). Cite the paper above if you use the data.
