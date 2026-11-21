# Documentation log – {{NAME}}

> Required by the protocol (same as GNN-CB): keep it updated during development and add it to your PR as
> `submissions/reports/{{NAME}}.md`. It is public. Organisers use it to audit budget and LLM use.

## 1. Summary
| Field | Value |
|---|---|
| Model type | {{MODEL_TYPE}} |
| Architecture | {{ARCH}} |
| Parameters | {{PARAMS}} |
| Seed | 25 |
| Hardware | {{HARDWARE}} |
| Total training time (all runs, min) | {{TOTAL_MIN}} (budget: 180 CPU-min) |
| Test inference time (s) | {{INFER_S}} (budget: 300 s) |
| Tuning runs | {{N_RUNS}} (budget: ≤ 5) |

## 2. Hyperparameter search strategy
<!-- What did you change between runs, and why? How did you choose the final run (validation only)? -->

## 3. All configurations tried
| Run | img size | width | lr | batch | epochs run | best epoch | minutes | val Dice |
|---|---|---|---|---|---|---|---|---|
{{RUN_ROWS}}

## 4. Loss-curve verification
<!-- Attach or link checkpoints/<run>_loss_curve.png for the final run. Did training converge? Did you use early stopping? -->

## 5. Pre/post-processing
<!-- Threshold, minimum-area filter, test-time augmentation, ... -->

## 6. LLM-assisted utility usage
<!-- human track: list every LLM-generated snippet (low-level utilities only, no task-specific design). Write "none" if none.
     human+llm / llm-only: name the model(s) and link the transcript. -->

## 7. Notes / what did not work
