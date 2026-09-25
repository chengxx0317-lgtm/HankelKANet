from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BUDGET_ORDER = {
    "0pct": 0,
    "1pct": 1,
    "5pct": 5,
    "10pct": 10,
    "20pct": 20,
    "100pct": 100,
}

MODE_ORDER = {
    "zero-shot": 0,
    "finetune": 1,
    "scratch": 2,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []

    for path in sorted(args.results_root.rglob("test_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["run_dir"] = str(path.parent)
        rows.append(data)

    if not rows:
        raise FileNotFoundError(
            f"No test_metrics.json found under {args.results_root}"
        )

    raw = pd.DataFrame(rows)
    raw_path = args.results_root / "all_runs.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    group_cols = ["model", "mode", "budget"]
    metrics = [
        "mae_db",
        "rmse_db",
        "r2",
        "macro_mae_db",
        "macro_rmse_db",
        "macro_r2",
    ]

    out_rows = []
    for keys, group in raw.groupby(group_cols, sort=False):
        row = {
            "model": keys[0],
            "mode": keys[1],
            "budget": keys[2],
            "runs": len(group),
        }

        for metric in metrics:
            if metric not in group:
                continue

            values = (
                pd.to_numeric(group[metric], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            row[f"{metric}_mean"] = (
                float(np.mean(values)) if len(values) else np.nan
            )
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )

        out_rows.append(row)

    summary = pd.DataFrame(out_rows)
    summary["_mode_order"] = summary["mode"].map(MODE_ORDER).fillna(99)
    summary["_budget_order"] = summary["budget"].map(BUDGET_ORDER).fillna(999)

    summary = (
        summary.sort_values(["_mode_order", "_budget_order"])
        .drop(columns=["_mode_order", "_budget_order"])
        .reset_index(drop=True)
    )

    output_path = args.output or (args.results_root / "summary_mean_std.csv")
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
