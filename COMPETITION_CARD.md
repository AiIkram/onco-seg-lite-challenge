# Competition card – ONCO-SEG Lite

*Specification in the style of GNN-CB Appendix A / Table 1, ready to drop into the benchmark paper.*

| Field | Value |
|---|---|
| **ID / name** | IMG-01 – ONCO-SEG Lite (breast tumour segmentation in ultrasound) · [github.com/AiIkram/onco-seg-lite-challenge](https://github.com/AiIkram/onco-seg-lite-challenge) |
| **Domain / category** | Medical imaging · 2D ultrasound · oncology |
| **Task type** | Binary semantic segmentation (lesion vs. background), including lesion-free images |
| **Source data** | BUSI (Al-Dhabyani et al., 2020): 780 B-mode breast ultrasound images (437 benign, 210 malignant, 133 normal) with expert masks |
| **Anti-contamination modification** | (i) Lesion-preserving random re-crop that changes the field of view and aspect ratio; (ii) gamma change plus multiplicative speckle-noise injection on every image; (iii) opaque HMAC IDs; (iv) near-duplicate-aware split. *Hard variant:* an extra simulated acquisition shift (gamma, strong speckle, blur) on the test set only (`--test-shift`). |
| **Split** | ~80/20, stratified by diagnosis, de-duplicated at group level; test masks hidden (encrypted) |
| **Input / output** | 256×256 grayscale PNG → 256×256 binary mask, RLE-encoded CSV (`image_id,rle`) |
| **Metric (ranking)** | Mean per-image Dice. Both masks empty → 1; exactly one empty → 0. |
| **Secondary metrics** | IoU, Dice on lesion images, specificity on normal images, detection sensitivity, Dice per class |
| **Constraints** | CPU only, 3 CPU-hours of training, ≤ 5 tuning runs, seed 25, ≤ 5M parameters, trained from scratch, inference ≤ 5 min |
| **Difficulty tier** | Medium (released configuration). The Hard variant adds `--test-shift`. As in GNN-CB, the tier is re-checked against the human score spread after the competition. |
| **Why it is hard** | Small dataset; ~17% lesion-free images, where any false-positive blob scores 0; low-contrast, speckled boundaries; the CPU budget limits model capacity and resolution |
| **Baseline** | TinyUNet (1,942,289 params), BCE + Dice, flip test-time augmentation, threshold and minimum-area filter tuned on validation, 128 px, CPU, seed 25. Its hidden-test score is the `organiser-baseline` row on the leaderboard, computed by `setup_competition.sh`. |
| **Submission** | Encrypted CSV + metadata JSON + documentation log via PR, scored by GitHub Actions, 1 final submission |
| **Designer** | Ikram Aissiou ([@AiIkram](https://github.com/AiIkram)). Excluded from competing in this competition. |
| **License** | Code MIT; data BUSI (cite Al-Dhabyani et al., 2020) |

## Paper notes

- **Mapping to GNN-CB metrics:** Dice is "higher is better", so the gap to Human Top is `score − human_top` with no sign correction (unlike MAE/NMAE).
- **Error taxonomy for LLM failures (suggested categories):** data loading/IO · mask resolution or RLE errors · empty-mask handling (false positives on normals) · CPU time-out · unsuitable architecture or capacity · training instability.
- **State the open protocol decisions** fixed in `llm_eval/README.md`: per-execution time limit, what happens after k failures, and sandbox contents.
