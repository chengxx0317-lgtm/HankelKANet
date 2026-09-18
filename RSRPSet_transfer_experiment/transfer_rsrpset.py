from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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
    p.add_argument("--mode", choices=("zero-shot", "finetune", "scratch"), required=True)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument(
        "--source-checkpoint", type=Path, default=None,
        help="Required for zero-shot and finetune; ignored for scratch."
    )
    p.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of the fixed TRAIN cell pool. Typical adaptation budgets: 0.01,0.05,0.10,0.20."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", type=Path, default=Path("./rsrpset_transfer_results"))

    # Keep architecture exactly aligned with the current main experiment.
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--phys-ch", type=int, default=32)
    p.add_argument("--z-min", type=float, default=math.pi * 1e-3)
    p.add_argument("--kappa-min", type=float, default=0.1)
    p.add_argument("--kappa-max", type=float, default=10.0)
    p.add_argument("--kappa-init", type=float, default=math.pi)

    # Training protocol. Defaults mirror the source experiment except that fine-tuning
    # uses a lower default LR to avoid destroying the transferred representation.
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--lr", type=float, default=None)
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
        # Keep deterministic=False because some operations used by the custom Hankel path
        # may not have deterministic kernels on every PyTorch/CUDA combination.


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


def torch_load_compat(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """Accept raw state_dicts and common checkpoint dictionary layouts."""
    if isinstance(obj, dict):
        # Raw state_dict: every value is tensor-like.
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
                    "Could not find model weights in checkpoint. Expected a raw state_dict "
                    "or one of keys: model/model_state_dict/state_dict/net."
                )
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(obj).__name__}")

    clean = {}
    for k, v in state.items():
        name = k[7:] if k.startswith("module.") else k
        clean[name] = v
    return clean


def load_model_weights_strict(model: nn.Module, checkpoint: Path) -> None:
    obj = torch_load_compat(checkpoint, map_location="cpu")
    state = extract_state_dict(obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is not strictly compatible with the current HKANNet.\n"
            f"Missing keys ({len(missing)}): {missing[:20]}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}"
        )


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


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=pred.dtype)
    denom = mask.sum()
    if denom.item() <= 0:
        raise RuntimeError("Batch contains no measured target grids.")
    return (((pred - target) ** 2) * mask).sum() / denom


class MetricAccumulator:
    """Micro/global measured-point metrics plus macro/per-cell metrics."""

    def __init__(self) -> None:
        self.n = 0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.y_sum = 0.0
        self.y2_sum = 0.0
        self.macro_mae = []
        self.macro_rmse = []
        self.macro_r2 = []
        self.cells = 0

    def update(self, pred_gain_db: torch.Tensor, true_gain_db: torch.Tensor, mask: torch.Tensor) -> None:
        # tensors: B,1,H,W
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
        if self.n == 0:
            raise RuntimeError("No measured points were accumulated.")
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
            "measured_points": int(self.n),
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
        # Compatibility with older PyTorch.
        return torch.cuda.amp.GradScaler(enabled=flag)


def make_loader(
    ds: RSRPSetMapDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
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

    result = metrics.result()
    result["masked_mse_norm"] = loss_num / max(loss_den, 1.0)
    return result


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
    seed_everything(args.seed)

    if args.mode in {"zero-shot", "finetune"} and args.source_checkpoint is None:
        raise ValueError(f"--source-checkpoint is required for mode={args.mode}")
    if args.mode == "zero-shot" and args.fraction != 1.0:
        # fraction has no meaning for zero-shot; avoid misleading metadata.
        print("[Info] --fraction is ignored in zero-shot mode.")
    if args.mode == "finetune" and not (0.0 < args.fraction <= 1.0):
        raise ValueError("Fine-tune fraction must be in (0,1].")
    if args.mode == "scratch" and not (0.0 < args.fraction <= 1.0):
        raise ValueError("Scratch fraction must be in (0,1].")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp_enabled = (not args.no_amp) and device.type == "cuda"

    fraction_tag = "0" if args.mode == "zero-shot" else f"{args.fraction:.4f}".rstrip("0").rstrip(".")
    run_dir = (
        args.output_root.resolve()
        / f"{args.mode}_frac{fraction_tag}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = None
    if args.mode != "zero-shot":
        train_ds = RSRPSetMapDataset(
            args.prepared_root,
            args.split_json,
            split="train",
            fraction=args.fraction,
            subset_seed=args.seed,
        )
        (run_dir / "train_cell_ids.txt").write_text(
            "\n".join(train_ds.cell_ids) + "\n", encoding="utf-8"
        )

    val_ds = RSRPSetMapDataset(args.prepared_root, args.split_json, split="val")
    test_ds = RSRPSetMapDataset(args.prepared_root, args.split_json, split="test")

    train_loader = None
    if train_ds is not None:
        train_loader = make_loader(
            train_ds, args.batch_size, shuffle=True, seed=args.seed,
            num_workers=args.num_workers
        )
    val_loader = make_loader(
        val_ds, args.batch_size, shuffle=False, seed=args.seed,
        num_workers=args.num_workers
    )
    test_loader = make_loader(
        test_ds, args.batch_size, shuffle=False, seed=args.seed,
        num_workers=args.num_workers
    )

    model = build_model(args)
    if args.mode in {"zero-shot", "finetune"}:
        source_ckpt = args.source_checkpoint.resolve()
        load_model_weights_strict(model, source_ckpt)
        print(f"Strictly loaded source checkpoint: {source_ckpt}")
    else:
        source_ckpt = None
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 88)
    print(f"Mode        : {args.mode}")
    print(f"Seed        : {args.seed}")
    print(f"Device      : {device}")
    print(f"AMP         : {amp_enabled}")
    print(f"Parameters  : {total_params:,} ({total_params/1e6:.6f} M)")
    print(f"Val cells   : {len(val_ds)}")
    print(f"Test cells  : {len(test_ds)}")
    if train_ds is not None:
        print(f"Train frac  : {args.fraction:.2%}")
        print(f"Train cells : {len(train_ds)}")
    print("=" * 88)

    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    config.update({
        "resolved_device": str(device),
        "amp_enabled": amp_enabled,
        "total_params": total_params,
        "train_cells": len(train_ds) if train_ds is not None else 0,
        "val_cells": len(val_ds),
        "test_cells": len(test_ds),
        "source_checkpoint_sha256": sha256_file(source_ckpt),
        "metric_mask": "measured outdoor receiver grids only",
        "metric_error_equivalence": "MAE/RMSE are identical in gain, PL, and RSRP dB because conversions differ only by sign/additive per-cell RSP",
        "db_normalization": {"min": DB_MIN, "max": DB_MAX, "clip": False},
    })
    save_json(run_dir / "config.json", config)

    # ---------- Zero-shot ----------
    if args.mode == "zero-shot":
        t0 = time.time()
        metrics = evaluate(model, test_loader, device, amp_enabled)
        metrics.update({
            "mode": args.mode,
            "fraction": 0.0,
            "seed": args.seed,
            "elapsed_sec": time.time() - t0,
        })
        save_json(run_dir / "test_metrics.json", metrics)
        print(json.dumps(metrics, indent=2))
        return

    # ---------- Fine-tune / scratch ----------
    assert train_loader is not None
    lr = args.lr
    if lr is None:
        lr = 4e-4 if args.mode == "finetune" else 4e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.eta_min
    )
    scaler = make_scaler(device, amp_enabled)

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.csv"

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

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Accumulate exact pixel-weighted normalized MSE.
            with torch.no_grad():
                diff2 = ((pred.float() - target.float()) ** 2) * mask.float()
                train_num += float(diff2.sum())
                train_den += float(mask.sum())

        val = evaluate(model, val_loader, device, amp_enabled)
        train_mse = train_num / max(train_den, 1.0)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_masked_mse_norm": train_mse,
            "val_masked_mse_norm": val["masked_mse_norm"],
            "val_mae_db": val["mae_db"],
            "val_rmse_db": val["rmse_db"],
            "val_r2": val["r2"],
            "lr": current_lr,
            "epoch_sec": time.time() - t_epoch,
        }
        write_csv_row(history_path, row)
        print(
            f"Epoch {epoch:03d} | trainMSE={train_mse:.6f} | "
            f"valMSE={val['masked_mse_norm']:.6f} | "
            f"MAE={val['mae_db']:.4f} dB | RMSE={val['rmse_db']:.4f} dB | "
            f"R2={val['r2']:.5f} | lr={current_lr:.3e}"
        )

        improved = val["masked_mse_norm"] < best_val - args.min_delta
        if improved:
            best_val = val["masked_mse_norm"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
            save_json(run_dir / "best_val_metrics.json", {"epoch": epoch, **val})
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.")
            break
        scheduler.step()

    if not best_path.exists():
        raise RuntimeError("No best model was saved; training did not complete a validation pass.")

    load_model_weights_strict(model, best_path)
    model = model.to(device)
    test = evaluate(model, test_loader, device, amp_enabled)
    test.update({
        "mode": args.mode,
        "fraction": args.fraction,
        "seed": args.seed,
        "train_cells": len(train_ds),
        "best_epoch": best_epoch,
        "best_val_masked_mse_norm": best_val,
        "learning_rate_initial": lr,
    })
    save_json(run_dir / "test_metrics.json", test)
    print("\nFinal test metrics:")
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    # Windows DataLoader compatibility.
    import torch.multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
