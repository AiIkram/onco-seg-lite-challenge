# LLM evaluation (GNN-CB protocol, image edition)

This folder runs LLMs on this competition under the **same protocol as GNN-CB** (§3.3):
a frozen zero-shot prompt, a plan-then-code answer, CPU-only execution and a bounded
execute-and-repair loop. Each model gets **one greedy run**.

| GNN-CB element | Here |
|---|---|
| Frozen prompt template, slots filled from the repo | `prompt_template.md` and `build_prompt.py`. Slots: `README`, `REPO_TREE`, `DATASET_SUMMARY`, `SUBMISSION_FORMAT`, `ALLOWED_LIBRARIES` (+ `TIME_BUDGET_MIN`) |
| `<plan>` then `<code>` (standalone `solution.py`) | parsed by `run_nonagent.py`; a missing `<code>` counts as a failed attempt |
| temperature 0, max_output_tokens 10K | `--temperature 0 --max-tokens 10000` (`--temperature -1` omits it for reasoning models that reject it) |
| CPU-only execution, no downloads | `CUDA_VISIBLE_DEVICES=""`; `--docker-image IMG` adds `--network none` |
| traceback appended, repair k ≤ 5 | `--repair-limit 5` (1 initial attempt + up to 5 repairs) |
| validity = schema + content check | `predictions.csv` must pass `competition.metrics.validate_submission` |
| non-agent orchestration script | `run_nonagent.py` |
| agent mode, same repair budget | `AGENT_PROTOCOL.md` |

## Decisions the GNN-CB text leaves open (fixed here, report them in the paper)

1. **Execution time limit:** 180 min per execution, matching the human 3 CPU-hour budget (`--exec-timeout-min`).
2. **After k failed repairs:** the run is **invalid**. It is logged in `results.csv` with `status=invalid`, counted in the **validity rate**, and not put on the leaderboard. For averaged comparisons, report it separately or as Dice = 0 and say which.
3. **Model sees** only `README.md`, `RULES.md`, `LICENSE`, `competition/`, `starter_code/` and `data/public/`, exactly what a human participant has. Nothing private is copied into the sandbox.
4. **Traceback length** returned to the model: the last 4,000 characters.

## Running

```bash
pip install -r starter_code/requirements.txt
python llm_eval/build_prompt.py                      # inspect llm_eval/prompt.md once and freeze it

# examples (one run per model)
ANTHROPIC_API_KEY=... python llm_eval/run_nonagent.py --provider anthropic --model claude-opus-4-6 --label llm_opus46
OPENAI_API_KEY=...    python llm_eval/run_nonagent.py --provider openai --model gpt-4.1 --label llm_gpt41
OPENAI_API_KEY=$TOGETHER_KEY python llm_eval/run_nonagent.py --provider openai \
    --base-url https://api.together.xyz/v1 --model meta-llama/Llama-3.3-70B-Instruct-Turbo --label llm_llama33
python llm_eval/run_nonagent.py --provider mock --label llm_mock_test   # dry run, no API key needed
```

A valid run writes `submissions/<label>.csv` together with metadata (`model_type: llm-only`) and a report. Submit it like any participant (encrypt, then open a PR from the LLM-arm account). An organiser can also score it offline with `competition/evaluate.py` and append the row to `docs/leaderboard.csv`.

## What to report (mirrors GNN-CB §4.2)

- Score per model and the **gap to Human Top** (shown on the leaderboard) and to the human average
- **Validity rate** and number of repairs needed (`results.csv`, `runs/<label>/summary.json`)
- Failure taxonomy from the transcripts, e.g. RLE/format errors, data reading, wrong resolution, empty-mask handling, CPU time-outs, unsuitable architecture
- Lesion Dice vs. specificity on normal scans: over-segmenting or false positives?
