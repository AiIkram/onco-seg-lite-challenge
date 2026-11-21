#!/usr/bin/env python3
"""
CI ONLY - decide whether a PR is a well-formed submission, WITHOUT executing
anything from the PR.

Input: TSV of "<status>\t<filename>" lines (from the GitHub PR files API).
Rules:
  * exactly ONE newly added file  submissions/encrypted/<name>.enc
  * optionally ONE submissions/metadata/<name>.json and ONE submissions/reports/<name>.md (same <name>)
  * nothing else may be touched (no code, no data, no workflows)
Writes to $GITHUB_OUTPUT: status, enc, name, meta  and a comment to argv[2] when rejected.
"""
import os
import re
import sys

ENC_RE = re.compile(r"^submissions/encrypted/([A-Za-z0-9_\-]{2,60})\.enc$")
META_RE = re.compile(r"^submissions/metadata/([A-Za-z0-9_\-]{2,60})\.json$")
REPORT_RE = re.compile(r"^submissions/reports/([A-Za-z0-9_\-]{2,60})\.md$")


def out(**kv):
    path = os.environ.get("GITHUB_OUTPUT")
    for k, v in kv.items():
        print(f"[output] {k}={v}")
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{k}={v}\n")


def reject(comment_path, status, msg):
    with open(comment_path, "w", encoding="utf-8") as f:
        f.write("## ❌ PR is not a valid submission\n\n" + msg + "\n\n"
                "A submission PR must add **only**:\n"
                "- `submissions/encrypted/<name>.enc` (exactly one, new file)\n"
                "- optionally `submissions/metadata/<name>.json` and `submissions/reports/<name>.md` (same `<name>`)\n\n"
                "This attempt does **not** count. Fix the PR (or open a new one) and push again.\n")
    out(status=status, enc="", name="", meta="", report="")
    sys.exit(0)


def main():
    files_tsv, comment_path = sys.argv[1], sys.argv[2]
    entries = []
    with open(files_tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                status, name = line.split("\t", 1)
                entries.append((status, name))

    encs, metas, reports, other = [], [], [], []
    for status, name in entries:
        if ENC_RE.match(name):
            encs.append((status, name))
        elif META_RE.match(name):
            metas.append((status, name))
        elif REPORT_RE.match(name):
            reports.append((status, name))
        else:
            other.append(name)

    if other:
        listed = "\n".join(f"- `{o}`" for o in other[:10])
        reject(comment_path, "bad_files", f"These files are not allowed in a submission PR:\n{listed}")
    if len(encs) != 1:
        reject(comment_path, "bad_files", f"Found {len(encs)} `.enc` files - exactly one is required.")
    status, enc = encs[0]
    if status != "added":
        reject(comment_path, "bad_files",
               f"`{enc}` must be a **new** file (status was `{status}`). Use a new file name for a new submission.")
    name = ENC_RE.match(enc).group(1)

    meta = ""
    if metas:
        if len(metas) > 1 or META_RE.match(metas[0][1]).group(1) != name or metas[0][0] == "removed":
            reject(comment_path, "bad_files", f"Metadata must be a single `submissions/metadata/{name}.json`.")
        meta = metas[0][1]

    report = ""
    if reports:
        if len(reports) > 1 or REPORT_RE.match(reports[0][1]).group(1) != name or reports[0][0] == "removed":
            reject(comment_path, "bad_files", f"The report must be a single `submissions/reports/{name}.md`.")
        report = reports[0][1]

    out(status="ok", enc=enc, name=name, meta=meta, report=report)


if __name__ == "__main__":
    main()
