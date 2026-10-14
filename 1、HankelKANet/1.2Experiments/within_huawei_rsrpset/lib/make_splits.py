from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


VERSION = "2.0-rsrpset-3804-fixed-nested-subsets"
EXPECTED_CELLS = 3804
TRAIN_COUNT = 2282
VAL_COUNT = 761
TEST_COUNT = 761
SUBSET_COUNTS = {
    "1pct": 23,
    "5pct": 114,
    "10pct": 228,
    "20pct": 456,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used once for the official train/val/test cell split.",
    )
    p.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Seed used once for the fixed nested TRAIN adaptation subsets.",
    )
    p.add_argument(
        "--expected-cells",
        type=int,
        default=EXPECTED_CELLS,
        help="Exact number of valid prepared cells expected. Set 0 to disable.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <prepared-root>/splits",
    )
    return p.parse_args()


def numeric_sort_key(text: str):
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def sha256_lines(values: Iterable[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_failed_cis(prepared_root: Path) -> List[str]:
    """Best-effort read of failed CI values for audit metadata only."""
    path = prepared_root / "preparation_errors.csv"
    if not path.exists():
        return []

    failed: List[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ci = row.get("CI") or row.get("ci") or row.get("cell_id")
                if ci is not None and str(ci).strip():
                    failed.append(str(ci).strip())
    except Exception:
        # The split itself must not depend on this optional audit file.
        return []
    return sorted(set(failed), key=numeric_sort_key)


def assert_unique(name: str, values: List[str]) -> None:
    if len(values) != len(set(values)):
        raise RuntimeError(f"Duplicate CI found inside {name}.")


def write_lines(path: Path, values: List[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepared = args.prepared_root.resolve()
    cell_dir = prepared / "cells"

    if not cell_dir.is_dir():
        raise FileNotFoundError(f"Prepared cell directory does not exist: {cell_dir}")

    ids = sorted((p.stem for p in cell_dir.glob("*.npz")), key=numeric_sort_key)
    if not ids:
        raise FileNotFoundError(f"No prepared .npz cells found in {cell_dir}")

    assert_unique("prepared cell list", ids)

    if args.expected_cells > 0 and len(ids) != args.expected_cells:
        raise RuntimeError(
            f"Expected {args.expected_cells} valid prepared cells but found {len(ids)}. "
            "Do not create the official split until the preparation result is understood."
        )

    if len(ids) != TRAIN_COUNT + VAL_COUNT + TEST_COUNT:
        raise RuntimeError(
            f"Official counts require {TRAIN_COUNT + VAL_COUNT + TEST_COUNT} cells, "
            f"but found {len(ids)}."
        )

    # Official fixed cell-level train/val/test split.
    split_rng = np.random.default_rng(args.split_seed)
    perm = np.asarray(ids, dtype=object)[split_rng.permutation(len(ids))].tolist()

    train = [str(x) for x in perm[:TRAIN_COUNT]]
    val = [str(x) for x in perm[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]]
    test = [str(x) for x in perm[TRAIN_COUNT + VAL_COUNT:]]

    assert_unique("train", train)
    assert_unique("val", val)
    assert_unique("test", test)

    train_set, val_set, test_set = set(train), set(val), set(test)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Data leakage detected: a CI appears in multiple splits.")
    if train_set | val_set | test_set != set(ids):
        raise RuntimeError("Split union does not exactly reproduce the prepared CI set.")

    # Fixed nested target-domain subsets. One ordering is generated once; every
    # percentage is a prefix of that same ordering.
    subset_rng = np.random.default_rng(args.subset_seed)
    train_perm = np.asarray(train, dtype=object)[subset_rng.permutation(len(train))].tolist()
    subsets: Dict[str, List[str]] = {
        name: [str(x) for x in train_perm[:count]]
        for name, count in SUBSET_COUNTS.items()
    }

    names = list(SUBSET_COUNTS.keys())
    for name in names:
        assert_unique(f"train subset {name}", subsets[name])
        if not set(subsets[name]).issubset(train_set):
            raise RuntimeError(f"Subset {name} contains CI outside the TRAIN split.")

    for smaller, larger in zip(names[:-1], names[1:]):
        if not set(subsets[smaller]).issubset(set(subsets[larger])):
            raise RuntimeError(f"Nested-subset invariant failed: {smaller} is not inside {larger}.")

    failed_cis = read_failed_cis(prepared)
    if set(failed_cis) & (train_set | val_set | test_set):
        raise RuntimeError("A CI listed in preparation_errors.csv appears in the official split.")

    out_dir = (args.output_dir or (prepared / "splits")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_json = out_dir / f"split_rsrpset3804_seed{args.split_seed}.json"
    payload = {
        "version": VERSION,
        "protocol": "fixed cell-level approximately 6:2:2 split on 3804 valid RSRPSet cells",
        "note": (
            "The original PEFNet paper reports a 2283/761/761 split for 3805 cells but does not "
            "publish the exact CI identities. After strict-outdoor preprocessing one CI has no valid "
            "target supervision, leaving 3804 valid cells. We therefore use 2282/761/761 with a fixed "
            "deterministic shuffle. The generated CI lists are frozen for all experiments."
        ),
        "prepared_root": str(prepared),
        "split_seed": int(args.split_seed),
        "subset_seed": int(args.subset_seed),
        "counts": {
            "all_valid": len(ids),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "adaptation_subset_counts": {
            k: len(v) for k, v in subsets.items()
        },
        "failed_preparation_cis_for_audit": failed_cis,
        "hashes": {
            "all_valid_sorted_sha256": sha256_lines(ids),
            "train_in_order_sha256": sha256_lines(train),
            "val_in_order_sha256": sha256_lines(val),
            "test_in_order_sha256": sha256_lines(test),
        },
        "train": train,
        "val": val,
        "test": test,
        "train_subsets": subsets,
    }
    split_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-readable frozen lists. These also make it easy to publish the exact CIs.
    write_lines(out_dir / "train_ci.txt", train)
    write_lines(out_dir / "val_ci.txt", val)
    write_lines(out_dir / "test_ci.txt", test)
    for name, values in subsets.items():
        write_lines(out_dir / f"train_{name}_ci.txt", values)

    audit = {
        "version": VERSION,
        "status": "PASS",
        "prepared_cell_count": len(ids),
        "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "split_disjoint": True,
        "split_union_equals_all_prepared_cells": True,
        "nested_subsets": True,
        "subset_counts": {k: len(v) for k, v in subsets.items()},
        "all_subsets_inside_train": True,
        "failed_cis_found_in_split": [],
        "split_seed": int(args.split_seed),
        "subset_seed": int(args.subset_seed),
        "official_split_json": str(split_json),
    }
    audit_path = out_dir / "split_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Split version : {VERSION}")
    print(f"Prepared cells: {len(ids)}")
    print(f"Split seed    : {args.split_seed}")
    print(f"Subset seed   : {args.subset_seed}")
    print(f"train/val/test: {len(train)} / {len(val)} / {len(test)}")
    print(
        "Fixed nested subsets: "
        + ", ".join(f"{name}={len(subsets[name])}" for name in SUBSET_COUNTS)
    )
    print("Split disjoint: PASS")
    print("Split union   : PASS")
    print("Nested subsets: PASS")
    if failed_cis:
        print(f"Failed preparation CIs excluded from split: {failed_cis}")
    print(f"Official split JSON: {split_json}")
    print(f"Audit report       : {audit_path}")


if __name__ == "__main__":
    main()
