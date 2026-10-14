from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage


REQUIRED_COLUMNS = {"CI", "RSP", "RSRP", "Hm", "UCI", "L", "cosA"}

SOURCE_DB_MIN = -147.0
SOURCE_DB_MAX = -47.84
PREPARATION_VERSION = "3.0-binary-building-nearest-strict-outdoor"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root", type=Path, required=True,
        help="Root containing the original DATASET_part*.pickle files.",
    )
    p.add_argument(
        "--output-root", type=Path, required=True,
        help="Destination directory for per-cell HankelKANet maps.",
    )
    p.add_argument(
        "--pattern", default="DATASET_part*.pickle",
        help="Recursive filename glob for source pickle files.",
    )
    p.add_argument("--grid-h", type=int, default=40)
    p.add_argument("--grid-w", type=int, default=80)
    p.add_argument("--dx", type=float, default=5.0, help="Grid spacing in metres.")
    p.add_argument(
        "--environment-fill",
        choices=("nearest",),
        default="nearest",
        help=(
            "How to complete unobserved binary building grids. Only 'nearest' "
            "is supported in v3 because UCI is categorical and should not be "
            "linearly interpolated."
        ),
    )
    p.add_argument(
        "--outdoor-rule",
        choices=("hm_lt5_and_uci_lt10",),
        default="hm_lt5_and_uci_lt10",
        help=(
            "Use only rows that are unambiguously non-building under the same "
            "rule used to reconstruct the building map: Hm < 5 m AND UCI < 10. "
            "This keeps target supervision consistent with the input building map."
        ),
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite already prepared cell files.",
    )
    p.add_argument(
        "--max-cells", type=int, default=0,
        help="Debug only: stop after this many successfully prepared/existing cells (0 = all).",
    )
    return p.parse_args()


def normalize_ci(value) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".")[0]
    return text


def fixed_r_map(grid_h: int, grid_w: int, dx: float) -> Tuple[np.ndarray, float]:
    """Tx is at grid index (0,0), matching the PEFNet origin convention."""
    row = np.arange(grid_h, dtype=np.float32)[:, None]
    col = np.arange(grid_w, dtype=np.float32)[None, :]
    d = np.sqrt((row * dx) ** 2 + (col * dx) ** 2).astype(np.float32)
    l_ref = math.sqrt(((grid_h - 1) * dx) ** 2 + ((grid_w - 1) * dx) ** 2)
    if l_ref <= 0:
        raise ValueError("Physical map diagonal must be positive.")
    return (d / l_ref).astype(np.float32), float(l_ref)


def reconstruct_xy(
    l_horizontal: np.ndarray,
    cos_alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    l_horizontal = np.asarray(l_horizontal, dtype=np.float64)
    cos_alpha = np.asarray(cos_alpha, dtype=np.float64)

    finite = cos_alpha[np.isfinite(cos_alpha)]
    if finite.size and (finite.min() < -1e-6 or finite.max() > 1.0 + 1e-6):
        raise ValueError(
            "cosA falls outside the expected [0,1] preprocessed range: "
            f"min={finite.min()}, max={finite.max()}"
        )

    cos_alpha = np.clip(cos_alpha, 0.0, 1.0)
    sin_alpha = np.sqrt(np.maximum(0.0, 1.0 - cos_alpha * cos_alpha))
    x = l_horizontal * sin_alpha
    y = l_horizontal * cos_alpha
    return x, y


def _grid_flat_index(rows: np.ndarray, cols: np.ndarray, grid_w: int) -> np.ndarray:
    return rows.astype(np.int64) * int(grid_w) + cols.astype(np.int64)


def _aggregate_grid_mean(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean value and observation count for duplicated samples in one grid."""
    flat = _grid_flat_index(rows, cols, grid_w)
    n = grid_h * grid_w
    sums = np.bincount(flat, weights=values.astype(np.float64), minlength=n)
    counts = np.bincount(flat, minlength=n)
    out = np.full(n, np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out.reshape(grid_h, grid_w), counts.reshape(grid_h, grid_w)


def _aggregate_grid_binary_any(
    rows: np.ndarray,
    cols: np.ndarray,
    binary_values: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    binary_values = np.asarray(binary_values, dtype=np.uint8)
    if not np.all((binary_values == 0) | (binary_values == 1)):
        raise ValueError("binary_values must contain only 0/1.")

    flat = _grid_flat_index(rows, cols, grid_w)
    n = grid_h * grid_w
    counts = np.bincount(flat, minlength=n)
    positives = np.bincount(flat, weights=binary_values.astype(np.float64), minlength=n)

    out = np.full(n, np.nan, dtype=np.float64)
    observed = counts > 0
    out[observed] = (positives[observed] > 0).astype(np.float64)

    conflicts = np.zeros(n, dtype=bool)
    conflicts[observed] = (positives[observed] > 0) & (positives[observed] < counts[observed])

    return (
        out.reshape(grid_h, grid_w),
        counts.reshape(grid_h, grid_w),
        conflicts.reshape(grid_h, grid_w),
    )


def _fill_binary_nearest(observed_binary: np.ndarray) -> np.ndarray:
    """Fill NaN binary grids using the nearest observed 0/1 grid label."""
    arr = np.asarray(observed_binary, dtype=np.float64)
    known = np.isfinite(arr)
    if not known.any():
        raise ValueError("No observed building/non-building grids are available.")
    if known.all():
        return arr.astype(np.float32)

    missing = ~known
    # For every missing cell, return the index of its nearest observed cell.
    indices = ndimage.distance_transform_edt(
        missing,
        return_distances=False,
        return_indices=True,
    )
    filled = arr[tuple(indices)]
    if not np.all(np.isfinite(filled)):
        raise RuntimeError("Nearest-neighbour building completion left NaN values.")
    if not np.all((filled == 0.0) | (filled == 1.0)):
        raise RuntimeError("Binary building completion produced non-binary values.")
    return filled.astype(np.float32)


def _validate_existing_output_version(out_root: Path, overwrite: bool) -> None:
    """Prevent accidental mixing of old and v2-prepared cells."""
    metadata_path = out_root / "metadata.json"
    cell_dir = out_root / "cells"
    if overwrite or not metadata_path.exists() or not cell_dir.exists():
        return
    try:
        old_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return
    old_version = old_meta.get("preparation_version")
    if old_version != PREPARATION_VERSION and any(cell_dir.glob("*.npz")):
        raise RuntimeError(
            "The output directory already contains cells prepared by a different "
            f"pipeline version ({old_version!r}). Current version is "
            f"{PREPARATION_VERSION!r}. Re-run with --overwrite or use a new "
            "output directory to avoid mixing incompatible maps."
        )


def prepare_one_cell(
    cell_df: pd.DataFrame,
    grid_h: int,
    grid_w: int,
    dx: float,
    r_map: np.ndarray,
    environment_fill: str,
    outdoor_rule: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    if environment_fill != "nearest":
        raise ValueError("v2 supports only nearest completion for binary building labels.")

    g = cell_df.copy()
    n_rows_raw = int(len(g))
    for col in ("RSP", "RSRP", "Hm", "UCI", "L", "cosA"):
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.dropna(subset=["RSP", "RSRP", "Hm", "UCI", "L", "cosA"])
    n_rows_complete = int(len(g))
    if n_rows_complete == 0:
        raise ValueError("Cell has no complete rows after numeric/NaN filtering.")

    rsp_unique = np.sort(g["RSP"].dropna().unique())
    if len(rsp_unique) != 1:
        raise ValueError(
            f"Expected one RSP per CI, found {len(rsp_unique)} values: {rsp_unique[:10]}"
        )
    rsp_dbm = float(rsp_unique[0])

    x, y = reconstruct_xy(g["L"].to_numpy(), g["cosA"].to_numpy())

    # PEFNet's 40x80 at 5 m corresponds to [0,200) x [0,400) in this convention.
    max_x = grid_h * dx
    max_y = grid_w * dx
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x < max_x)
        & (y >= 0.0)
        & (y < max_y)
    )
    if not inside.any():
        raise ValueError("No receiver rows fall inside the configured map domain.")

    g = g.iloc[np.nonzero(inside)[0]].copy()
    x = x[inside]
    y = y[inside]
    rows = np.floor(x / dx).astype(np.int64)
    cols = np.floor(y / dx).astype(np.int64)
    rows = np.clip(rows, 0, grid_h - 1)
    cols = np.clip(cols, 0, grid_w - 1)

    hm = g["Hm"].to_numpy(dtype=np.float64)
    uci = g["UCI"].to_numpy(dtype=np.float64)

    # Classify OBSERVED rows first. Do not interpolate raw Hm/UCI values.
    row_building = ((hm >= 5.0) | (uci >= 10.0)).astype(np.uint8)
    building_observed, env_count, env_label_conflict = _aggregate_grid_binary_any(
        rows,
        cols,
        row_building,
        grid_h,
        grid_w,
    )
    environment_observed_mask = np.isfinite(building_observed)
    building = _fill_binary_nearest(building_observed)

    # Measured target mask. Use the exact complement of the published building
    # rule at the row level so the target definition and building semantics agree.
    if outdoor_rule != "hm_lt5_and_uci_lt10":
        raise ValueError(
            "This v3 requires outdoor_rule='hm_lt5_and_uci_lt10'."
        )
    outdoor = (hm < 5.0) & (uci < 10.0)

    target_rows = rows[outdoor]
    target_cols = cols[outdoor]
    gain_values = (
        g.loc[outdoor, "RSRP"].to_numpy(dtype=np.float64)
        - g.loc[outdoor, "RSP"].to_numpy(dtype=np.float64)
    )
    if gain_values.size == 0:
        raise ValueError("Cell has no unambiguously outdoor measured target rows.")

    gain_db, target_count = _aggregate_grid_mean(
        target_rows,
        target_cols,
        gain_values,
        grid_h,
        grid_w,
    )

    candidate_target_mask = np.isfinite(gain_db)
    candidate_target_on_building = candidate_target_mask & (building >= 0.5)
    measured_mask = candidate_target_mask & (building < 0.5)
    if not measured_mask.any():
        raise ValueError(
            "Cell has no valid measured target grids after building-consistency filtering."
        )

    target_norm = np.zeros((grid_h, grid_w), dtype=np.float32)
    scale = SOURCE_DB_MAX - SOURCE_DB_MIN
    target_norm[measured_mask] = (
        (gain_db[measured_mask] - SOURCE_DB_MIN) / scale
    ).astype(np.float32)

    gain_db_filled = np.zeros((grid_h, grid_w), dtype=np.float32)
    gain_db_filled[measured_mask] = gain_db[measured_mask].astype(np.float32)

    # RSRP is retained for diagnostics only; training/evaluation can be derived from gain + RSP.
    rsrp_dbm_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    rsrp_dbm_map[measured_mask] = (gain_db[measured_mask] + rsp_dbm).astype(np.float32)

    # Diagnostics / quality controls.
    observed_building = np.zeros((grid_h, grid_w), dtype=np.uint8)
    observed_building[environment_observed_mask] = building_observed[
        environment_observed_mask
    ].astype(np.uint8)


    target_on_building = measured_mask & (building >= 0.5)
    if target_on_building.any():
        raise RuntimeError(
            "Internal error: final measured_mask overlaps reconstructed building grids."
        )

    measured_count = int(measured_mask.sum())
    candidate_count = int(candidate_target_mask.sum())
    excluded_building_count = int(candidate_target_on_building.sum())
    out_of_low = measured_mask & (gain_db_filled < SOURCE_DB_MIN)
    out_of_high = measured_mask & (gain_db_filled > SOURCE_DB_MAX)

    arrays: Dict[str, np.ndarray] = {
        "building": building.astype(np.float32),
        "building_observed": observed_building,
        "r_map": r_map.astype(np.float32),
        "target_norm": target_norm,
        "gain_db": gain_db_filled,
        "rsrp_dbm": rsrp_dbm_map,
        "measured_mask": measured_mask.astype(np.uint8),
        "target_count": target_count.astype(np.int32),
        "environment_observed_mask": environment_observed_mask.astype(np.uint8),
        "environment_label_conflict_mask": env_label_conflict.astype(np.uint8),
        "candidate_target_mask": candidate_target_mask.astype(np.uint8),
        "excluded_target_on_building_mask": candidate_target_on_building.astype(np.uint8),
        "target_on_building_mask": target_on_building.astype(np.uint8),
        "rsp_dbm": np.asarray(rsp_dbm, dtype=np.float32),
    }

    n_inside = int(inside.sum())
    n_env_grids = int(environment_observed_mask.sum())
    n_observed_building = int(
        (observed_building.astype(bool) & environment_observed_mask).sum()
    )
    n_env_conflict = int(env_label_conflict.sum())
    n_target_on_building = int(target_on_building.sum())
    n_target_multi = int(((target_count > 1) & measured_mask).sum())

    gain_measured = gain_db[measured_mask]
    stats = {
        "rows_total": n_rows_raw,
        "rows_complete": n_rows_complete,
        "rows_inside_domain": n_inside,
        "inside_ratio_complete": n_inside / max(n_rows_complete, 1),
        "inside_ratio_raw": n_inside / max(n_rows_raw, 1),
        "environment_observed_grids": n_env_grids,
        "environment_grid_coverage": n_env_grids / float(grid_h * grid_w),
        "observed_building_grids": n_observed_building,
        "observed_building_fraction": n_observed_building / max(n_env_grids, 1),
        "environment_label_conflict_grids": n_env_conflict,
        "environment_label_conflict_fraction": n_env_conflict / max(n_env_grids, 1),
        "building_fraction": float(building.mean()),
        "measured_outdoor_rows": int(outdoor.sum()),
        "candidate_target_grids": candidate_count,
        "candidate_target_grid_coverage": candidate_count / float(grid_h * grid_w),
        "excluded_target_on_building_grids": excluded_building_count,
        "excluded_target_on_building_fraction": excluded_building_count / max(candidate_count, 1),
        "measured_outdoor_grids": measured_count,
        "measured_grid_coverage": measured_count / float(grid_h * grid_w),
        "multi_measurement_target_grids": n_target_multi,
        "multi_measurement_target_fraction": n_target_multi / max(measured_count, 1),
        "target_on_building_grids": n_target_on_building,
        "target_on_building_fraction": n_target_on_building / max(measured_count, 1),
        "rsp_dbm": rsp_dbm,
        "gain_db_min": float(np.nanmin(gain_measured)),
        "gain_db_max": float(np.nanmax(gain_measured)),
        "gain_below_source_range_grids": int(out_of_low.sum()),
        "gain_above_source_range_grids": int(out_of_high.sum()),
        "gain_outside_source_range_fraction": (
            int(out_of_low.sum()) + int(out_of_high.sum())
        ) / max(measured_count, 1),
    }
    return arrays, stats


def _write_dataset_quality_report(
    summary: pd.DataFrame,
    errors: pd.DataFrame,
    out_root: Path,
    grid_h: int,
    grid_w: int,
) -> dict:
    """Create human-readable dataset-level checks after preparation."""
    report: Dict[str, object] = {
        "preparation_version": PREPARATION_VERSION,
        "prepared_or_existing_cells": int(len(summary)),
        "failed_cells": int(len(errors)),
        "grid_shape": [int(grid_h), int(grid_w)],
    }

    if summary.empty:
        (out_root / "dataset_quality_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return report

    def numeric(name: str) -> pd.Series:
        if name not in summary.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(summary[name], errors="coerce").dropna()

    coverage = numeric("measured_grid_coverage")
    env_cov = numeric("environment_grid_coverage")
    bld = numeric("building_fraction")
    obs_bld = numeric("observed_building_fraction")
    conflicts = numeric("target_on_building_fraction")
    excluded_target = numeric("excluded_target_on_building_fraction")
    env_conflicts = numeric("environment_label_conflict_fraction")
    outside = numeric("gain_outside_source_range_fraction")
    inside = numeric("inside_ratio_complete")

    def describe(s: pd.Series) -> dict:
        if s.empty:
            return {}
        return {
            "count": int(s.size),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
            "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        }

    report.update({
        "measured_grid_coverage": describe(coverage),
        "environment_grid_coverage": describe(env_cov),
        "completed_building_fraction": describe(bld),
        "observed_building_fraction": describe(obs_bld),
        "target_on_building_fraction": describe(conflicts),
        "excluded_target_on_building_fraction": describe(excluded_target),
        "environment_label_conflict_fraction": describe(env_conflicts),
        "gain_outside_source_range_fraction": describe(outside),
        "inside_ratio_complete": describe(inside),
    })

    if not bld.empty:
        report["cells_building_fraction_lt_1pct"] = int((bld < 0.01).sum())
        report["cells_building_fraction_gt_90pct"] = int((bld > 0.90).sum())
    if not coverage.empty:
        report["cells_measured_coverage_lt_10pct"] = int((coverage < 0.10).sum())
        report["cells_measured_coverage_gt_90pct"] = int((coverage > 0.90).sum())
    if not outside.empty:
        report["cells_with_any_gain_outside_source_range"] = int((outside > 0).sum())
    if not conflicts.empty:
        report["cells_with_target_on_building_conflict"] = int((conflicts > 0).sum())

    (out_root / "dataset_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    out_root = args.output_root.resolve()
    cell_dir = out_root / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    _validate_existing_output_version(out_root, args.overwrite)

    source_files = sorted(raw_root.rglob(args.pattern))
    if not source_files:
        raise FileNotFoundError(f"No files matching {args.pattern!r} under {raw_root}")

    r_map, l_ref = fixed_r_map(args.grid_h, args.grid_w, args.dx)
    seen = set()
    summary_rows = []
    error_rows = []
    processed = 0

    print(f"Preparation version: {PREPARATION_VERSION}")
    print(f"Found {len(source_files)} raw pickle files.")
    print(
        f"Output map: {args.grid_h}x{args.grid_w}, dx={args.dx:g} m, "
        f"L_ref={l_ref:.6f} m"
    )
    print(
        "Building reconstruction: observed row classification "
        "(Hm>=5 OR UCI>=10) -> grid aggregation -> nearest binary completion"
    )
    print(f"Outdoor target rule: {args.outdoor_rule}")

    for file_idx, path in enumerate(source_files, 1):
        print(f"[{file_idx}/{len(source_files)}] Loading {path}")
        df = pd.read_pickle(path)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{path} is {type(df).__name__}, expected pandas.DataFrame")
        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise KeyError(f"{path} misses required columns: {missing}")

        for ci_raw, group in df.groupby("CI", sort=False):
            ci = normalize_ci(ci_raw)
            if ci in seen:
                raise RuntimeError(
                    f"CI {ci} appears in more than one group/source file. "
                    "This preparation script expects each CI to be unique globally."
                )
            seen.add(ci)
            cell_path = cell_dir / f"{ci}.npz"

            if cell_path.exists() and not args.overwrite:
                # Existing v3 cell. Keep it, but summary cannot reconstruct every
                # raw-row statistic without rereading/repreparing the cell.
                with np.load(cell_path) as pack:
                    mask_count = int(pack["measured_mask"].sum())
                    building_fraction = float(pack["building"].mean())
                    rsp_dbm = float(pack["rsp_dbm"])
                    target_on_building = int(
                        pack["target_on_building_mask"].sum()
                    ) if "target_on_building_mask" in pack.files else np.nan
                summary_rows.append({
                    "CI": ci,
                    "source_file": str(path),
                    "status": "existing",
                    "measured_outdoor_grids": mask_count,
                    "measured_grid_coverage": mask_count / float(args.grid_h * args.grid_w),
                    "building_fraction": building_fraction,
                    "target_on_building_grids": target_on_building,
                    "target_on_building_fraction": (
                        target_on_building / max(mask_count, 1)
                        if np.isfinite(target_on_building) else np.nan
                    ),
                    "rsp_dbm": rsp_dbm,
                })
                processed += 1
            else:
                try:
                    arrays, stats = prepare_one_cell(
                        group,
                        grid_h=args.grid_h,
                        grid_w=args.grid_w,
                        dx=args.dx,
                        r_map=r_map,
                        environment_fill=args.environment_fill,
                        outdoor_rule=args.outdoor_rule,
                    )
                    np.savez_compressed(cell_path, **arrays)
                    summary_rows.append({
                        "CI": ci,
                        "source_file": str(path),
                        "status": "prepared",
                        **stats,
                    })
                    processed += 1
                except Exception as exc:
                    error_rows.append({
                        "CI": ci,
                        "source_file": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"  [WARN] CI={ci}: {type(exc).__name__}: {exc}")

            if args.max_cells > 0 and processed >= args.max_cells:
                break
        if args.max_cells > 0 and processed >= args.max_cells:
            break

    summary = pd.DataFrame(summary_rows)
    errors = pd.DataFrame(error_rows)
    summary.to_csv(out_root / "cell_summary.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(out_root / "preparation_errors.csv", index=False, encoding="utf-8-sig")

    quality = _write_dataset_quality_report(
        summary,
        errors,
        out_root,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
    )

    metadata = {
        "preparation_version": PREPARATION_VERSION,
        "raw_root": str(raw_root),
        "source_files": [str(p) for p in source_files],
        "grid_h": args.grid_h,
        "grid_w": args.grid_w,
        "dx_m": args.dx,
        "physical_extent_nominal_m": [args.grid_h * args.dx, args.grid_w * args.dx],
        "L_ref_m": l_ref,
        "tx_grid_index": [0, 0],
        "coordinate_reconstruction": (
            "x=L*sqrt(1-cosA^2), y=L*cosA; cosA expected in [0,1]"
        ),
        "building_rule": "observed row is building iff Hm >= 5 m OR UCI >= 10",
        "building_grid_aggregation": "binary-any within each 5 m grid",
        "building_reconstruction": (
            "nearest-neighbour completion of observed binary building labels; "
            "raw Hm/UCI are not spatially interpolated"
        ),
        "environment_fill": args.environment_fill,
        "outdoor_target_rule": (
            "Hm < 5 m AND UCI < 10 at row level; after 5 m grid aggregation, "
            "candidate target grids labelled building are excluded from supervision"
        ),
        "target_definition": "gain_dB = RSRP - RSP = -PL",
        "target_normalization": {
            "db_min": SOURCE_DB_MIN,
            "db_max": SOURCE_DB_MAX,
            "clip": False,
            "reason": "must remain identical to the existing RadioMapSeer checkpoint scale",
        },
        "target_supervision": (
            "measured, building-consistent outdoor grids only; no interpolated "
            "RSRP/PL pseudo-labels"
        ),
        "prepared_cells": int(len(summary_rows)),
        "failed_cells": int(len(error_rows)),
        "quality_report": "dataset_quality_report.json",
    }
    (out_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nPreparation finished.")
    print(f"Prepared/existing cells: {len(summary_rows)}")
    print(f"Failed cells:            {len(error_rows)}")
    print(f"Cell files:              {cell_dir}")
    print(f"Summary:                 {out_root / 'cell_summary.csv'}")
    print(f"Quality report:          {out_root / 'dataset_quality_report.json'}")

    cov = quality.get("measured_grid_coverage", {})
    bld = quality.get("completed_building_fraction", {})
    conflict = quality.get("target_on_building_fraction", {})
    if cov:
        print(
            "Mean measured-grid coverage: "
            f"{100.0 * float(cov.get('mean', float('nan'))):.2f}%"
        )
    if bld:
        print(
            "Mean completed-building fraction: "
            f"{100.0 * float(bld.get('mean', float('nan'))):.2f}%"
        )
    if conflict:
        print(
            "Mean target-on-building conflict fraction: "
            f"{100.0 * float(conflict.get('mean', float('nan'))):.2f}%"
        )


if __name__ == "__main__":
    main()