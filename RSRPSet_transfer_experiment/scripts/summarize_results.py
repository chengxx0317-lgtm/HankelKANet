from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
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
        raise FileNotFoundError(f"No test_metrics.json under {args.results_root}")

    raw = pd.DataFrame(rows)
    raw_path = args.results_root / "all_runs.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    group_cols = ["mode", "fraction"]
    metrics = ["mae_db", "rmse_db", "r2", "macro_mae_db", "macro_rmse_db", "macro_r2"]
    out_rows = []
    for keys, g in raw.groupby(group_cols, sort=True):
        row = {"mode": keys[0], "fraction": keys[1], "runs": len(g)}
        for m in metrics:
            if m in g:
                vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(dtype=float)
                row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
                row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out_rows.append(row)
    summary = pd.DataFrame(out_rows)
    out = args.output or (args.results_root / "summary_mean_std.csv")
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nRaw runs: {raw_path}")
    print(f"Summary : {out}")


if __name__ == "__main__":
    main()
