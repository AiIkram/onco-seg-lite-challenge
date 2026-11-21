#!/usr/bin/env python3
"""
Non-agent LLM evaluation (GNN-CB 3.3: frozen prompt -> plan-then-code -> CPU execution
-> bounded execute-and-repair). One greedy run per model.

    # OpenAI-compatible endpoint (OpenAI, Together, DeepSeek, Kimi, vLLM, Ollama, ...)
    OPENAI_API_KEY=... python llm_eval/run_nonagent.py --provider openai --model gpt-4.1 \
        --base-url https://api.openai.com/v1 --label llm_gpt41

    # Anthropic
    ANTHROPIC_API_KEY=... python llm_eval/run_nonagent.py --provider anthropic --model claude-opus-4-6 --label llm_opus46

    # Gemini via its OpenAI-compatible endpoint
    OPENAI_API_KEY=$GEMINI_KEY python llm_eval/run_nonagent.py --provider openai --model gemini-2.5-pro \
        --base-url https://generativelanguage.googleapis.com/v1beta/openai --label llm_gemini25pro

Loop (per run): call LLM -> extract <plan>/<code> -> write solution.py in a sandbox copy of the
public repo -> `python solution.py` on CPU (CUDA hidden) with a timeout -> validity check
(predictions.csv passes competition.metrics.validate_submission). On failure the traceback /
validation errors are appended to the conversation and the model repairs. At most
--repair-limit repairs (default 5) after the initial attempt.

Outputs in llm_eval/runs/<label>/:
    transcript.jsonl  every message + execution log      summary.json  status, counts, timings
    attempt_<k>/solution.py, stdout.txt                   predictions.csv (if valid)
and, if valid, submissions/<label>.csv + metadata + report, ready for encrypt_submission.py.
A run that never produces a valid file is recorded as INVALID in llm_eval/results.csv
(it counts against the validity rate; it is not placed on the leaderboard).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "llm_eval"))
import pandas as pd  # noqa: E402
from build_prompt import PUBLIC_ITEMS, build  # noqa: E402
from competition.metrics import read_submission_csv, validate_submission  # noqa: E402

PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.S | re.I)
CODE_RE = re.compile(r"<code>(.*?)</code>", re.S | re.I)
FENCE_RE = re.compile(r"^```(?:python)?\s*\n(.*?)\n```\s*$", re.S)


# --------------------------------------------------------------------------- #
# Providers (plain HTTP, no SDK dependency)                                    #
# --------------------------------------------------------------------------- #
def _post(url, headers, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def call_llm(args, messages):
    if args.provider == "mock":
        k = sum(1 for m in messages if m["role"] == "assistant")
        files = sorted(os.listdir(args.mock_dir))
        with open(os.path.join(args.mock_dir, files[min(k, len(files) - 1)]), encoding="utf-8") as f:
            return f.read(), {}
    if args.provider == "openai":
        payload = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens}
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        r = _post(args.base_url.rstrip("/") + "/chat/completions",
                  {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                  payload)
        return r["choices"][0]["message"]["content"] or "", r.get("usage", {})
    if args.provider == "anthropic":
        payload = {"model": args.model, "max_tokens": args.max_tokens, "messages": messages}
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        r = _post("https://api.anthropic.com/v1/messages",
                  {"Content-Type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                   "anthropic-version": "2023-06-01"}, payload)
        text = "".join(b.get("text", "") for b in r["content"] if b.get("type") == "text")
        return text, r.get("usage", {})
    raise ValueError(args.provider)


# --------------------------------------------------------------------------- #
# Sandbox + execution                                                          #
# --------------------------------------------------------------------------- #
def make_sandbox(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    ignore = shutil.ignore_patterns("checkpoints", "runs_log.csv", "__pycache__", "*.pt")
    for item in PUBLIC_ITEMS:
        src, dst = os.path.join(ROOT, item), os.path.join(path, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=ignore)
        elif os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def execute(sandbox, timeout_s, docker_image=None, cpus=None):
    pred = os.path.join(sandbox, "predictions.csv")
    if os.path.exists(pred):
        os.remove(pred)
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="25")
    if docker_image:  # strongest isolation: no network, fixed CPUs
        cmd = ["docker", "run", "--rm", "--network", "none", "-v", f"{os.path.abspath(sandbox)}:/work",
               "-w", "/work"] + (["--cpus", str(cpus)] if cpus else []) + [docker_image, "python", "solution.py"]
    else:
        cmd = [sys.executable, "solution.py"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=sandbox, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        rc, out = p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = "timeout"
        out = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + \
              f"\nTIMEOUT: solution.py exceeded {timeout_s // 60} minutes and was killed."
    return rc, out, time.time() - t0


def check_validity(sandbox, test_ids):
    pred = os.path.join(sandbox, "predictions.csv")
    if not os.path.exists(pred):
        return ["predictions.csv was not created in the repository root."]
    try:
        return validate_submission(read_submission_csv(pred), test_ids)
    except Exception as e:  # noqa: BLE001
        return [f"predictions.csv could not be read: {e}"]


def extract(text):
    plan, code = PLAN_RE.search(text), CODE_RE.search(text)
    if not code:
        return (plan.group(1).strip() if plan else ""), None
    c = code.group(1).strip()
    m = FENCE_RE.match(c)
    return (plan.group(1).strip() if plan else ""), (m.group(1) if m else c)


def repair_message(kind, detail, tail_chars):
    detail = detail[-tail_chars:]
    return (f"Your solution failed ({kind}).\n\n```\n{detail}\n```\n\n"
            "Fix the problem. Answer again in the same format: <plan>...</plan> then the COMPLETE solution.py "
            "inside <code>...</code>.")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic", "mock"], required=True)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--label", required=True, help="run / leaderboard name, e.g. llm_opus46")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--temperature", type=float, default=0.0, help="use --temperature -1 to omit (reasoning models)")
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--repair-limit", type=int, default=5)
    ap.add_argument("--exec-timeout-min", type=float, default=180, help="per execution (= human 3 CPU-hour budget)")
    ap.add_argument("--tail-chars", type=int, default=4000, help="traceback characters returned to the model")
    ap.add_argument("--docker-image", default=None, help="run solution.py in docker --network none")
    ap.add_argument("--cpus", type=float, default=None)
    ap.add_argument("--mock-dir", default=os.path.join(ROOT, "llm_eval", "mock_responses"))
    args = ap.parse_args()
    if args.temperature is not None and args.temperature < 0:
        args.temperature = None

    run_dir = os.path.join(ROOT, "llm_eval", "runs", args.label)
    os.makedirs(run_dir, exist_ok=True)
    sandbox = os.path.join(run_dir, "sandbox")
    make_sandbox(sandbox)
    test_ids = pd.read_csv(os.path.join(ROOT, "data", "public", "test_ids.csv"), dtype=str).image_id

    prompt = build(int(args.exec_timeout_min))
    with open(os.path.join(run_dir, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(prompt)
    messages = [{"role": "user", "content": prompt}]
    log = open(os.path.join(run_dir, "transcript.jsonl"), "w", encoding="utf-8")
    attempts, status, t_start = [], "invalid", time.time()

    for k in range(args.repair_limit + 1):
        t0 = time.time()
        text, usage = call_llm(args, messages)
        messages.append({"role": "assistant", "content": text})
        log.write(json.dumps({"attempt": k, "role": "assistant", "content": text, "usage": usage,
                              "llm_seconds": round(time.time() - t0, 1)}) + "\n")
        plan, code = extract(text)
        a_dir = os.path.join(run_dir, f"attempt_{k}")
        os.makedirs(a_dir, exist_ok=True)
        rec = {"attempt": k, "llm_seconds": round(time.time() - t0, 1), "usage": usage}

        if code is None:
            rec.update(outcome="format_error")
            fb = repair_message("no <code>...</code> block found", "Your answer did not contain <code>...</code>.",
                                args.tail_chars)
        else:
            with open(os.path.join(a_dir, "solution.py"), "w", encoding="utf-8") as f:
                f.write(code)
            shutil.copy2(os.path.join(a_dir, "solution.py"), os.path.join(sandbox, "solution.py"))
            rc, out, secs = execute(sandbox, int(args.exec_timeout_min * 60), args.docker_image, args.cpus)
            with open(os.path.join(a_dir, "stdout.txt"), "w", encoding="utf-8") as f:
                f.write(out)
            errors = check_validity(sandbox, test_ids) if rc == 0 else []
            rec.update(returncode=rc, exec_seconds=round(secs, 1))
            if rc == 0 and not errors:
                rec.update(outcome="valid")
                attempts.append(rec)
                log.write(json.dumps({"attempt": k, "role": "system", "event": rec}) + "\n")
                status = "valid"
                shutil.copy2(os.path.join(sandbox, "predictions.csv"), os.path.join(run_dir, "predictions.csv"))
                break
            if rc != 0:
                rec.update(outcome="runtime_error" if rc != "timeout" else "timeout")
                fb = repair_message("execution error" if rc != "timeout" else "time limit", out, args.tail_chars)
            else:
                rec.update(outcome="invalid_output")
                fb = repair_message("invalid predictions.csv", "\n".join(errors), args.tail_chars)
        attempts.append(rec)
        log.write(json.dumps({"attempt": k, "role": "system", "event": rec}) + "\n")
        if k < args.repair_limit:
            messages.append({"role": "user", "content": fb})
            log.write(json.dumps({"attempt": k, "role": "user", "content": fb}) + "\n")
    log.close()

    summary = {
        "label": args.label, "provider": args.provider, "model": args.model, "status": status,
        "n_llm_calls": len(attempts), "n_repairs": max(0, len(attempts) - 1),
        "repair_limit": args.repair_limit, "temperature": args.temperature, "max_tokens": args.max_tokens,
        "total_minutes": round((time.time() - t_start) / 60, 1),
        "exec_minutes": round(sum(a.get("exec_seconds", 0) for a in attempts) / 60, 1),
        "outcomes": [a["outcome"] for a in attempts],
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    res = os.path.join(ROOT, "llm_eval", "results.csv")
    row = pd.DataFrame([{k: summary[k] for k in ("label", "provider", "model", "status", "n_llm_calls",
                                                  "n_repairs", "exec_minutes", "finished")}])
    row.to_csv(res, mode="a", header=not os.path.exists(res), index=False)

    if status == "valid":
        sub_dir = os.path.join(ROOT, "submissions")
        os.makedirs(os.path.join(sub_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(sub_dir, "reports"), exist_ok=True)
        shutil.copy2(os.path.join(run_dir, "predictions.csv"), os.path.join(sub_dir, f"{args.label}.csv"))
        with open(os.path.join(sub_dir, "metadata", f"{args.label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": "llm-only", "architecture": "see report", "params": "",
                       "llm": args.model, "seed": 25, "n_tuning_runs": 1,
                       "train_minutes_total": summary["exec_minutes"],
                       "hardware": "CPU sandbox (non-agent harness)"}, f, indent=2)
        with open(os.path.join(sub_dir, "reports", f"{args.label}.md"), "w", encoding="utf-8") as f:
            f.write(f"# Documentation log – {args.label}\n\nNon-agent LLM run (frozen prompt, plan-then-code, "
                    f"execute-and-repair k≤{args.repair_limit}).\n\n```json\n{json.dumps(summary, indent=2)}\n```\n\n"
                    f"## Final plan\n\n{plan}\n")
        print(f"VALID after {summary['n_repairs']} repair(s). Submit with:\n"
              f"  python encryption/encrypt_submission.py submissions/{args.label}.csv")
    else:
        print(f"INVALID after {summary['n_llm_calls']} attempts: {summary['outcomes']}")
    print(f"Logs: {os.path.relpath(run_dir, ROOT)}/")


if __name__ == "__main__":
    main()
