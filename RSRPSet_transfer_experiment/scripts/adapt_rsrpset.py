from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules import HKANNet
from rsrpset_dataset import RSRPSetMapDataset


DB_MIN = -147.0
DB_MAX = -47.84
DB_SCALE = DB_MAX - DB_MIN

EXPECTED_SUBSET_COUNTS = {
    "1pct": 23,
    "5pct": 114,
    "10pct": 228,
    "20pct": 456,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("finetune", "scratch"), required=True)
    p.add_argument("--budget", choices=("1pct", "5pct", "10pct", "20pct", "100pct"), required=True)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument(
        "--source-checkpoint", type=Path, default=None,
        help="Required for finetune; ignored for scratch."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", type=Path, default=Path("./rsrpset_adaptation_results"))

    # Current final HankelKANet architecture.
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--phys-ch", type=int, default=32)
    p.add_argument("--z-min", type=float, default=math.pi * 1e-3)
    p.add_argument("--kappa-min", type=float, default=0.1)
    p.add_argument("--kappa-max", type=float, default=10.0)
    p.add_argument("--kappa-init", type=float, default=math.pi)

    # Training.
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=250)
    p.add_argument(
        "--val-interval", type=int, default=5,
        help="Run full 761-cell validation every N epochs. Epoch 1 and final epoch are also validated."
    )
    p.add_argument(
        "--patience-epochs", type=int, default=50,
        help="Early-stop after this many epochs without a validation improvement."
    )
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument(
        "--lr", type=float, default=None,
        help="Default: 4e-4 for finetune, 4e-3 for scratch."
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sha256_file(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids: List[str]) -> str:
    payload = "\n".join(ids) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def torch_load_compat(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            state = obj
        else:
            state = None
            for key in ("model", "model_state_dict", "state_dict", "net"):
                candidate = obj.get(key)
                if isinstance(candidate, dict) and candidate:
                    state = candidate
                    break
            if state is None:
                raise KeyError(
                    "Could not find model weights. Expected raw state_dict or "
                    "one of model/model_state_dict/state_dict/net."
                )
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(obj).__name__}")

    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }


def load_model_weights_strict(model: nn.Module, checkpoint: Path) -> None:
    state = extract_state_dict(torch_load_compat(checkpoint, map_location="cpu"))
    model.load_state_dict(state, strict=True)


def build_model(args: argparse.Namespace) -> HKANNet:
    return HKANNet(
        base_ch=args.base_ch,
        phys_ch=args.phys_ch,
        learnable_k=True,
        z_min=args.z_min,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        kappa_init=args.kappa_init,
    )


def get_frozen_train_ids(split_json: Path, budget: str) -> tuple[List[str], Dict[str, Any]]:
    payload = json.loads(split_json.read_text(encoding="utf-8"))

    train_ids = [str(x) for x in payload["train"]]
    if budget == "100pct":
        ids = train_ids
    else:
        subsets = payload.get("train_subsets")
        if not isinstance(subsets, dict) or budget not in subsets:
            raise KeyError(
                f"{split_json} does not contain train_subsets[{budget!r}]. "
                "Use the split generated by make_splits_v2.py."
            )
        ids = [str(x) for x in subsets[budget]]

        expected = EXPECTED_SUBSET_COUNTS[budget]
        if len(ids) != expected:
            raise RuntimeError(
                f"Frozen {budget} subset has {len(ids)} cells, expected {expected}."
            )

    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate CI detected in frozen {budget} subset.")

    train_set = set(train_ids)
    outside = [x for x in ids if x not in train_set]
    if outside:
        raise RuntimeError(f"Frozen subset contains CI outside train split: {outside[:10]}")

    # Cross-check the intended nested/frozen protocol when metadata are available.
    if payload.get("subset_seed") not in (None, 42):
        print(f"[Warning] split_json subset_seed={payload.get('subset_seed')} (expected 42).")

    return ids, payload


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(dtype=pred.dtype)
    denom = m.sum()
    if denom.item() <= 0:
        raise RuntimeError("Batch contains no valid measured grids.")
    return ((((pred - target) ** 2) * m).sum() / denom)


class MetricAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.y_sum = 0.0
        self.y2_sum = 0.0
        self.macro_mae: List[float] = []
        self.macro_rmse: List[float] = []
        self.macro_r2: List[float] = []
        self.cells = 0

    def update(self, pred_gain_db: torch.Tensor, true_gain_db: torch.Tensor, mask: torch.Tensor) -> None:
        b = pred_gain_db.shape[0]
        for i in range(b):
            valid = mask[i, 0] > 0.5
            if not torch.any(valid):
                continue

            p = pred_gain_db[i, 0][valid].detach().double().cpu()
            y = true_gain_db[i, 0][valid].detach().double().cpu()
            e = p - y

            n_i = int(y.numel())
            self.n += n_i
            self.abs_sum += float(e.abs().sum())
            self.sq_sum += float((e * e).sum())
            self.y_sum += float(y.sum())
            self.y2_sum += float((y * y).sum())
            self.cells += 1

            mae_i = float(e.abs().mean())
            rmse_i = float(torch.sqrt((e * e).mean()))
            sse_i = float((e * e).sum())
            y_mean = float(y.mean())
            sst_i = float(((y - y_mean) ** 2).sum())
            r2_i = 1.0 - sse_i / sst_i if sst_i > 1e-12 else float("nan")

            self.macro_mae.append(mae_i)
            self.macro_rmse.append(rmse_i)
            if np.isfinite(r2_i):
                self.macro_r2.append(r2_i)

    def result(self) -> Dict[str, float]:
        if self.n <= 0:
            raise RuntimeError("No measured grids accumulated.")

        mae = self.abs_sum / self.n
        rmse = math.sqrt(self.sq_sum / self.n)
        sst = self.y2_sum - (self.y_sum * self.y_sum) / self.n
        r2 = 1.0 - self.sq_sum / sst if sst > 1e-12 else float("nan")

        return {
            "mae_db": mae,
            "rmse_db": rmse,
            "r2": r2,
            "macro_mae_db": float(np.mean(self.macro_mae)),
            "macro_rmse_db": float(np.mean(self.macro_rmse)),
            "macro_r2": float(np.mean(self.macro_r2)) if self.macro_r2 else float("nan"),
            "measured_grids": int(self.n),
            "valid_cells": int(self.cells),
        }


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler(device: torch.device, enabled: bool):
    flag = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=flag)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=flag)


def make_loader(ds, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=gen if shuffle else None,
        persistent_workers=(num_workers > 0),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool) -> Dict[str, float]:
    model.eval()
    loss_num = 0.0
    loss_den = 0.0
    metrics = MetricAccumulator()

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        target = batch["target_norm"].to(device, non_blocking=True)
        true_gain = batch["gain_db"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with amp_context(device, amp_enabled):
            pred = model(x)

        diff2 = ((pred.float() - target.float()) ** 2) * mask.float()
        loss_num += float(diff2.sum())
        loss_den += float(mask.sum())

        pred_gain = pred.float() * DB_SCALE + DB_MIN
        metrics.update(pred_gain, true_gain.float(), mask.float())

    out = metrics.result()
    out["masked_mse_norm"] = loss_num / max(loss_den, 1.0)
    return out


def write_csv_row(path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.mode == "finetune" and args.source_checkpoint is None:
        raise ValueError("--source-checkpoint is required for finetune mode.")
    if args.mode == "finetune" and args.budget == "100pct":
        raise ValueError("The planned protocol uses 100% only for the scratch upper-bound.")
    if args.val_interval <= 0:
        raise ValueError("--val-interval must be >= 1.")
    if args.patience_epochs <= 0:
        raise ValueError("--patience-epochs must be >= 1.")

    seed_everything(args.seed)

    prepared_root = args.prepared_root.resolve()
    split_json = args.split_json.resolve()
    frozen_train_ids, split_payload = get_frozen_train_ids(split_json, args.budget)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp_enabled = (not args.no_amp) and device.type == "cuda"

    # Explicit IDs ensure data identity is frozen and does not depend on training seed.
    train_ds = RSRPSetMapDataset(
        prepared_root,
        split_json,
        split="train",
        fraction=1.0,
        cell_ids=frozen_train_ids,
    )
    val_ds = RSRPSetMapDataset(prepared_root, split_json, split="val")
    test_ds = RSRPSetMapDataset(prepared_root, split_json, split="test")

    # Strong anti-leakage audit.
    train_set = set(train_ds.cell_ids)
    val_set = set(val_ds.cell_ids)
    test_set = set(test_ds.cell_ids)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Data leakage detected: train/val/test CI overlap.")

    train_loader = make_loader(
        train_ds, args.batch_size, True, args.seed, args.num_workers
    )
    val_loader = make_loader(
        val_ds, args.batch_size, False, args.seed, args.num_workers
    )
    test_loader = make_loader(
        test_ds, args.batch_size, False, args.seed, args.num_workers
    )

    model = build_model(args)

    source_ckpt = None
    if args.mode == "finetune":
        source_ckpt = args.source_checkpoint.resolve()
        load_model_weights_strict(model, source_ckpt)
        checkpoint_status = "PASS"
    else:
        checkpoint_status = "N/A (scratch)"

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # Shape probe before any optimization.
    probe = next(iter(train_loader))
    probe_x = probe["input"].to(device)
    model.eval()
    with torch.no_grad():
        with amp_context(device, amp_enabled):
            probe_y = model(probe_x)
    if tuple(probe_y.shape) != (probe_x.shape[0], 1, 40, 80):
        raise RuntimeError(
            f"Unexpected model output shape {tuple(probe_y.shape)} for "
            f"input {tuple(probe_x.shape)}."
        )

    run_dir = (
        args.output_root.resolve()
        / f"{args.mode}_{args.budget}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "train_cell_ids.txt").write_text(
        "\n".join(train_ds.cell_ids) + "\n", encoding="utf-8"
    )

    lr = args.lr
    if lr is None:
        lr = 4e-4 if args.mode == "finetune" else 4e-3

    config = {
        "mode": args.mode,
        "budget": args.budget,
        "seed": args.seed,
        "prepared_root": str(prepared_root),
        "split_json": str(split_json),
        "split_version": split_payload.get("version"),
        "split_seed": split_payload.get("split_seed"),
        "subset_seed": split_payload.get("subset_seed"),
        "frozen_subset_source": f"split_json.train_subsets.{args.budget}" if args.budget != "100pct" else "split_json.train",
        "frozen_train_cell_count": len(train_ds),
        "frozen_train_ids_sha256": sha256_ids(train_ds.cell_ids),
        "val_cells": len(val_ds),
        "test_cells": len(test_ds),
        "training_seed_only_changes_stochasticity": True,
        "source_checkpoint": str(source_ckpt) if source_ckpt else None,
        "source_checkpoint_sha256": sha256_file(source_ckpt),
        "checkpoint_strict_load": checkpoint_status,
        "model_parameters": total_params,
        "probe_input_shape": list(probe_x.shape),
        "probe_output_shape": list(probe_y.shape),
        "optimizer": "AdamW",
        "lr_initial": lr,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "max_epochs": args.max_epochs,
        "eta_min": args.eta_min,
        "batch_size": args.batch_size,
        "val_interval_epochs": args.val_interval,
        "early_stopping_patience_epochs": args.patience_epochs,
        "grad_clip": args.grad_clip,
        "amp": amp_enabled,
        "metric_mask": "measured strict-outdoor grids only",
        "db_normalization": {
            "min": DB_MIN,
            "max": DB_MAX,
            "clip": False,
        },
        "test_usage": "evaluated once after validation-based checkpoint selection",
    }
    save_json(run_dir / "config.json", config)

    print("=" * 92)
    print(f"Mode                    : {args.mode}")
    print(f"Budget                  : {args.budget}")
    print(f"Training seed           : {args.seed}")
    print(f"Frozen subset seed      : {split_payload.get('subset_seed')}")
    print(f"Train cells             : {len(train_ds)}")
    print(f"Val cells               : {len(val_ds)}")
    print(f"Test cells              : {len(test_ds)}")
    print(f"Subset CI hash          : {sha256_ids(train_ds.cell_ids)}")
    print(f"Checkpoint strict load  : {checkpoint_status}")
    print(f"Parameters              : {total_params:,} ({total_params/1e6:.6f} M)")
    print(f"Input probe             : {tuple(probe_x.shape)}")
    print(f"Output probe            : {tuple(probe_y.shape)}")
    print(f"Initial LR              : {lr:.3e}")
    print(f"Validation interval     : every {args.val_interval} epochs")
    print(f"Patience                : {args.patience_epochs} epochs")
    print(f"Target update           : ALL model parameters")
    print("=" * 92)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.eta_min
    )
    scaler = make_scaler(device, amp_enabled)

    # Initial validation is useful for finetune: it is the source model before target update.
    t_init = time.time()
    initial_val = evaluate(model, val_loader, device, amp_enabled)
    initial_val["epoch"] = 0
    initial_val["elapsed_sec"] = time.time() - t_init
    save_json(run_dir / "initial_val_metrics.json", initial_val)
    print(
        f"Initial val | MAE={initial_val['mae_db']:.4f} dB | "
        f"RMSE={initial_val['rmse_db']:.4f} dB | R2={initial_val['r2']:.5f}"
    )

    best_val = initial_val["masked_mse_norm"]
    best_epoch = 0
    last_improve_epoch = 0
    best_path = run_dir / "best_model.pt"
    torch.save(model.state_dict(), best_path)
    save_json(run_dir / "best_val_metrics.json", initial_val)

    history_path = run_dir / "history.csv"
    total_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t_epoch = time.time()
        train_num = 0.0
        train_den = 0.0

        for batch in train_loader:
            x = batch["input"].to(device, non_blocking=True)
            target = batch["target_norm"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_context(device, amp_enabled):
                pred = model(x)
                loss = masked_mse(pred, target, mask)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}: {float(loss.detach().cpu())}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                diff2 = ((pred.float() - target.float()) ** 2) * mask.float()
                train_num += float(diff2.sum())
                train_den += float(mask.sum())

        train_mse = train_num / max(train_den, 1.0)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        do_val = (
            epoch == 1
            or epoch % args.val_interval == 0
            or epoch == args.max_epochs
        )

        row = {
            "epoch": epoch,
            "train_masked_mse_norm": train_mse,
            "lr": current_lr,
            "epoch_train_sec": time.time() - t_epoch,
            "validated": int(do_val),
            "val_masked_mse_norm": "",
            "val_mae_db": "",
            "val_rmse_db": "",
            "val_r2": "",
        }

        if do_val:
            t_val = time.time()
            val = evaluate(model, val_loader, device, amp_enabled)
            val_sec = time.time() - t_val

            row.update({
                "val_masked_mse_norm": val["masked_mse_norm"],
                "val_mae_db": val["mae_db"],
                "val_rmse_db": val["rmse_db"],
                "val_r2": val["r2"],
            })

            print(
                f"Epoch {epoch:03d} | trainMSE={train_mse:.6f} | "
                f"valMSE={val['masked_mse_norm']:.6f} | "
                f"MAE={val['mae_db']:.4f} dB | RMSE={val['rmse_db']:.4f} dB | "
                f"R2={val['r2']:.5f} | lr={current_lr:.3e} | val={val_sec:.1f}s"
            )

            if val["masked_mse_norm"] < best_val - args.min_delta:
                best_val = val["masked_mse_norm"]
                best_epoch = epoch
                last_improve_epoch = epoch
                torch.save(model.state_dict(), best_path)
                save_json(run_dir / "best_val_metrics.json", {"epoch": epoch, **val})
        else:
            print(
                f"Epoch {epoch:03d} | trainMSE={train_mse:.6f} | "
                f"lr={current_lr:.3e}"
            )

        write_csv_row(history_path, row)

        if epoch - last_improve_epoch >= args.patience_epochs:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best validation epoch={best_epoch}."
            )
            break

    # Final test: exactly once, after validation model selection.
    load_model_weights_strict(model, best_path)
    model = model.to(device)

    t_test = time.time()
    test = evaluate(model, test_loader, device, amp_enabled)
    test_sec = time.time() - t_test

    test.update({
        "mode": args.mode,
        "budget": args.budget,
        "seed": args.seed,
        "train_cells": len(train_ds),
        "frozen_train_ids_sha256": sha256_ids(train_ds.cell_ids),
        "best_epoch": best_epoch,
        "best_val_masked_mse_norm": best_val,
        "learning_rate_initial": lr,
        "test_elapsed_sec": test_sec,
        "total_elapsed_sec": time.time() - total_start,
    })
    save_json(run_dir / "test_metrics.json", test)

    print("\nFinal test metrics")
    print(json.dumps(test, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
