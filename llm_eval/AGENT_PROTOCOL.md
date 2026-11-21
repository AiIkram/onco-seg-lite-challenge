# Agent-mode protocol (Claude Code, Codex CLI, …)

GNN-CB §3.3.4: the agent inspects the repository, writes the solution, runs it and repairs failures
**without human intervention**, under the **same repair budget** as non-agent runs.

## Setup (operator)
1. Make a fresh **public-only** workspace containing only what participants have:
   ```bash
   python -c "import sys; sys.path.insert(0,'llm_eval'); from run_nonagent import make_sandbox; make_sandbox('agent_ws')"
   ```
   The workspace has `README.md`, `RULES.md`, `competition/`, `starter_code/` and `data/public/`. It has no keys, no `data/private*` and no git history.
2. Use a CPU-only machine or container without a GPU (`CUDA_VISIBLE_DEVICES=""`). Disable network access for executed code if the agent allows it; at minimum, forbid downloads in the task text below.
3. Generate the frozen prompt: `python llm_eval/build_prompt.py --out agent_ws/TASK.md`

## Task message (paste verbatim as the agent's only instruction)

```
Read TASK.md in this directory and solve the competition it describes.
Write your solution to solution.py, run it with `python solution.py`, and make sure it writes a valid
predictions.csv in this directory (check it with: python starter_code/check_submission.py predictions.csv).
Rules: CPU only, no internet or downloads, seed 25, follow README.md and RULES.md.
You may execute solution.py at most 6 times in total (1 initial run + at most 5 repair runs).
Before your final answer, output your plan inside <plan>...</plan>.
```

## During the run
- Do not intervene: no hints, no answers to clarifying questions beyond "proceed with your best judgement".
- Record the session (the agent's transcript/export) and count executions of `solution.py`. If the agent runs it more than 6 times, record that as a protocol deviation.
- Use the agent's default (native) session budget for tokens and time, as in GNN-CB, and record it.

## After the run
```bash
cp agent_ws/predictions.csv submissions/llm_<agent>.csv
python starter_code/check_submission.py submissions/llm_<agent>.csv
```
Write `submissions/metadata/llm_<agent>.json` with `"model_type": "llm-only"` and `"llm": "<agent + model>"`. Save the transcript in `llm_eval/runs/llm_<agent>/`, then submit like any participant. If no valid `predictions.csv` exists after the budget, record the run as `invalid` in `llm_eval/results.csv`.
