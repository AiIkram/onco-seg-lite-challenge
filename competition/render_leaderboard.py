#!/usr/bin/env python3
"""
Render docs/leaderboard.csv (single source of truth) into:
    leaderboard/leaderboard.md   - static view on GitHub (auto-generated, don't edit)
    docs/leaderboard.json        - convenience copy for external tools

Ranking: best submission per participant, by mean Dice (desc); ties broken by
earlier date; tied scores share a rank (1, 2, 2, 4).
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "docs", "leaderboard.csv")
MD = os.path.join(ROOT, "leaderboard", "leaderboard.md")
JS = os.path.join(ROOT, "docs", "leaderboard.json")


def best_per_team(df):
    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df = df.sort_values(["score", "date"], ascending=[False, True])
    df["n_submissions"] = df.groupby("team")["team"].transform("size")
    best = df.drop_duplicates("team", keep="first").reset_index(drop=True)
    best["rank"] = best["score"].rank(method="min", ascending=False).astype(int)
    return best


def main():
    os.makedirs(os.path.dirname(MD), exist_ok=True)
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False) if os.path.exists(CSV) else pd.DataFrame()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    with open(MD, "w", encoding="utf-8") as f:
        f.write("# 🏆 ONCO-SEG Lite - Leaderboard\n\n")
        f.write(f"_Auto-generated from `docs/leaderboard.csv` - last updated {now}. Do not edit by hand._\n\n")
        if df.empty:
            f.write("No submissions yet.\n")
            best = pd.DataFrame()
        else:
            best = best_per_team(df)
            humans = best[best["model_type"] == "human"]["score"]
            human_top = humans.max() if len(humans) else None
            best["gap"] = (best["score"] - human_top).map(lambda g: f"{g:+.4f}") if human_top is not None else "-"
            f.write("| Rank | Participant | Mean Dice | Δ Human Top | IoU | Lesion Dice | Specificity | Type | Architecture | Params | Date |\n")
            f.write("|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|\n")
            for _, r in best.iterrows():
                medal = {1: "🥇 ", 2: "🥈 ", 3: "🥉 "}.get(r["rank"], "")
                params = f"{int(r['params']):,}" if str(r.get("params", "")).isdigit() else "-"
                f.write(f"| {medal}{r['rank']} | {r['team']} | **{r['score']:.4f}** | {r['gap']} | {r['iou']} | "
                        f"{r['lesion_dice']} | {r['specificity']} | {r['model_type']} | "
                        f"{r.get('architecture', '') or '-'} | {params} | {r['date']} |\n")
            f.write(f"\n{len(df)} scored submission(s) from {len(best)} participant(s). "
                    f"Δ Human Top = score − best `human` score"
                    f"{f' ({human_top:.4f})' if human_top is not None else ' (no human entry yet)'}.\n")

    with open(JS, "w", encoding="utf-8") as f:
        json.dump({"last_updated": now,
                   "entries": [] if best.empty else best.to_dict(orient="records")}, f, indent=2)
    print(f"Rendered {os.path.relpath(MD, ROOT)} and {os.path.relpath(JS, ROOT)}")


if __name__ == "__main__":
    main()
