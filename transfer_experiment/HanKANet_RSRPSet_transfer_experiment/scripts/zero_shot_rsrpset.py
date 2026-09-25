from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules import HKANNet
from rsrpset_dataset import RSRPSetMapDataset

DB_MIN = -147.0
DB_MAX = -47.84
DB_SCALE = DB_MAX - DB_MIN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("./rsrpset_zero_shot"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_load_weights(path: Path):
    # Prefer safe weights-only loading when supported by the local PyTorch.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        state = obj
    elif isinstance(obj, dict):
        state = None
        for key in ("model", "model_state_dict", "state_dict", "net"):
            v = obj.get(key)
            if isinstance(v, dict) and v:
                state = v
                break
        if state is None:
            raise KeyError("No model state_dict found in checkpoint.")
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(obj).__name__}")

    clean: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        clean[k[7:] if k.startswith("module.") else k] = v
    return clean


def build_model() -> HKANNet:
    return HKANNet(
        base_ch=32,
        phys_ch=32,
        learnable_k=True,
        z_min=math.pi * 1e-3,
        kappa_min=0.1,
        kappa_max=10.0,
        kappa_init=math.pi,
    )


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def write_per_cell(path: Path, rows):
    fieldnames = [
        "ci", "measured_points", "mae_db", "rmse_db", "r2",
        "true_gain_mean_db", "pred_gain_mean_db",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool):
    model.eval()

    n_total = 0
    abs_sum = 0.0
    sq_sum = 0.0
    y_sum = 0.0
    y2_sum = 0.0
    norm_sq_sum = 0.0
    per_cell = []

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        target_norm = batch["target_norm"].to(device, non_blocking=True)
        true_gain = batch["gain_db"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        cis = batch["ci"]

        with amp_context(device, amp_enabled):
            pred_norm = model(x)

        # Shape guard: must preserve native Huawei spatial size.
        if pred_norm.shape != target_norm.shape:
            raise RuntimeError(
                f"Output/target shape mismatch: pred={tuple(pred_norm.shape)}, "
                f"target={tuple(target_norm.shape)}"
            )

        pred_norm_f = pred_norm.float()
        target_norm_f = target_norm.float()
        true_gain_f = true_gain.float()
        mask_f = mask.float()

        pred_gain = pred_norm_f * DB_SCALE + DB_MIN
        norm_sq_sum += float((((pred_norm_f - target_norm_f) ** 2) * mask_f).sum().cpu())

        for i, ci in enumerate(cis):
            valid = mask_f[i, 0] > 0.5
            if not bool(torch.any(valid)):
                raise RuntimeError(f"Test cell {ci} contains no measured target grids.")

            p = pred_gain[i, 0][valid].double().cpu()
            y = true_gain_f[i, 0][valid].double().cpu()
            e = p - y
            n = int(y.numel())

            n_total += n
            abs_sum += float(e.abs().sum())
            sq_sum += float((e * e).sum())
            y_sum += float(y.sum())
            y2_sum += float((y * y).sum())

            sse = float((e * e).sum())
            y_mean = float(y.mean())
            sst = float(((y - y_mean) ** 2).sum())
            r2 = 1.0 - sse / sst if sst > 1e-12 else float("nan")

            per_cell.append({
                "ci": str(ci),
                "measured_points": n,
                "mae_db": float(e.abs().mean()),
                "rmse_db": float(torch.sqrt((e * e).mean())),
                "r2": r2,
                "true_gain_mean_db": float(y.mean()),
                "pred_gain_mean_db": float(p.mean()),
            })

    if n_total <= 0:
        raise RuntimeError("No measured test grids were evaluated.")

    global_mae = abs_sum / n_total
    global_rmse = math.sqrt(sq_sum / n_total)
    global_sst = y2_sum - (y_sum * y_sum) / n_total
    global_r2 = 1.0 - sq_sum / global_sst if global_sst > 1e-12 else float("nan")

    macro_mae = float(np.mean([r["mae_db"] for r in per_cell]))
    macro_rmse = float(np.mean([r["rmse_db"] for r in per_cell]))
    valid_r2 = [r["r2"] for r in per_cell if np.isfinite(r["r2"])]
    macro_r2 = float(np.mean(valid_r2)) if valid_r2 else float("nan")

    metrics = {
        "mae_db": global_mae,
        "rmse_db": global_rmse,
        "r2": global_r2,
        "macro_mae_db": macro_mae,
        "macro_rmse_db": macro_rmse,
        "macro_r2": macro_r2,
        "measured_test_grids": n_total,
        "test_cells": len(per_cell),
        "masked_mse_norm": norm_sq_sum / n_total,
    }
    return metrics, per_cell


def main() -> None:
    args = parse_args()
    prepared_root = args.prepared_root.resolve()
    split_json = args.split_json.resolve()
    source_ckpt = args.source_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_ckpt.exists():
        raise FileNotFoundError(source_ckpt)
    if not split_json.exists():
        raise FileNotFoundError(split_json)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp_enabled = (not args.no_amp) and device.type == "cuda"

    # IMPORTANT: zero-shot sees test only. No target train/val set is instantiated.
    test_ds = RSRPSetMapDataset(
        prepared_root=prepared_root,
        split_json=split_json,
        split="test",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model()
    state = extract_state_dict(torch_load_weights(source_ckpt))
    # True strict loading: any missing/unexpected key or size mismatch raises immediately.
    model.load_state_dict(state, strict=True)

    total_params = sum(p.numel() for p in model.parameters())
    if total_params != 14_952_353:
        raise RuntimeError(f"Unexpected parameter count: {total_params:,}")

    model = model.to(device)

    # One-batch interface check before the full run.
    probe = next(iter(test_loader))
    probe_x = probe["input"].to(device, non_blocking=True)
    with torch.no_grad(), amp_context(device, amp_enabled):
        probe_y = model(probe_x)
    expected = (probe_x.shape[0], 1, probe_x.shape[2], probe_x.shape[3])
    if tuple(probe_y.shape) != expected:
        raise RuntimeError(f"Interface check failed: got {tuple(probe_y.shape)}, expected {expected}")

    print("=" * 88)
    print("Huawei RSRPSet zero-shot evaluation")
    print(f"Checkpoint strict load : PASS")
    print(f"Parameters             : {total_params:,} ({total_params/1e6:.6f} M)")
    print(f"Test cells             : {len(test_ds)}")
    print(f"Input probe            : {tuple(probe_x.shape)}")
    print(f"Output probe           : {tuple(probe_y.shape)}")
    print(f"Device / AMP           : {device} / {amp_enabled}")
    print("Metric mask            : measured outdoor grids only")
    print("Target update          : NONE")
    print("=" * 88)

    t0 = time.time()
    metrics, per_cell = evaluate(model, test_loader, device, amp_enabled)
    elapsed = time.time() - t0
    metrics["elapsed_sec"] = elapsed

    metadata = {
        "protocol": "zero-shot synthetic-to-real; RadioMapSeer checkpoint -> frozen Huawei test split",
        "prepared_root": str(prepared_root),
        "split_json": str(split_json),
        "split_json_sha256": sha256_file(split_json),
        "source_checkpoint": str(source_ckpt),
        "source_checkpoint_sha256": sha256_file(source_ckpt),
        "strict_load": True,
        "total_params": total_params,
        "architecture": {
            "base_ch": 32,
            "phys_ch": 32,
            "orders": [0, 1],
            "z_min": math.pi * 1e-3,
            "kappa_min": 0.1,
            "kappa_max": 10.0,
            "kappa_init": math.pi,
            "input_channels": ["building", "r_map"],
        },
        "db_normalization": {"min": DB_MIN, "max": DB_MAX, "clip": False},
        "metric_mask": "measured outdoor grids only",
        "metric_primary": "global/micro aggregation over all measured test grids",
        "target_domain_parameter_updates": 0,
        "test_cells": len(test_ds),
        "device": str(device),
        "amp": amp_enabled,
        "metrics": metrics,
    }

    (output_dir / "zero_shot_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "zero_shot_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_per_cell(output_dir / "zero_shot_per_cell.csv", per_cell)

    print("\nFinal zero-shot test metrics")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
