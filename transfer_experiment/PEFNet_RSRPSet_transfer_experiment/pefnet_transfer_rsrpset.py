from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pefnet_model import PhysicsInformedModel
from pefnet_rsrpset_dataset import PEFNetRSRPSetDataset


DB_MIN = -147.0
DB_MAX = -47.84
DB_SCALE = DB_MAX - DB_MIN

EXPECTED_SUBSET_COUNTS = {
    "1pct": 23,
    "5pct": 114,
    "10pct": 228,
    "20pct": 456,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("zero-shot", "finetune", "scratch"), required=True)
    p.add_argument(
        "--budget",
        choices=("0pct", "1pct", "5pct", "10pct", "20pct", "100pct"),
        required=True,
    )
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", type=Path, default=Path("./pefnet_rsrpset_results"))

    p.add_argument("--target-frequency-ghz", type=float, default=2.6)
    p.add_argument("--target-dx-m", type=float, default=5.0)
    p.add_argument("--tx-row", type=int, default=0)
    p.add_argument("--tx-col", type=int, default=0)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=250)
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--patience-epochs", type=int, default=50)
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-data", type=float, default=1.0)
    p.add_argument("--lambda-phy", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sha256_file(path: Optional[Path]):
    if path is None:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids):
    payload = "\n".join(ids) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def torch_load_compat(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(obj: Any):
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        state = obj
    elif isinstance(obj, dict):
        state = None
        for key in ("model", "model_state_dict", "state_dict", "net"):
            candidate = obj.get(key)
            if isinstance(candidate, dict) and candidate:
                state = candidate
                break
        if state is None:
            raise KeyError("No model state_dict found in checkpoint.")
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(obj).__name__}")

    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def load_model_weights_strict(model: nn.Module, checkpoint: Path):
    state = extract_state_dict(torch_load_compat(checkpoint, map_location="cpu"))
    model.load_state_dict(state, strict=True)


def get_frozen_train_ids(split_json: Path, budget: str):
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    train_ids = [str(x) for x in payload["train"]]

    if budget == "100pct":
        ids = train_ids
    else:
        subsets = payload.get("train_subsets")
        if not isinstance(subsets, dict) or budget not in subsets:
            raise KeyError(f"Missing frozen train_subsets[{budget!r}] in split JSON.")
        ids = [str(x) for x in subsets[budget]]
        expected = EXPECTED_SUBSET_COUNTS[budget]
        if len(ids) != expected:
            raise RuntimeError(
                f"Frozen {budget} subset has {len(ids)} cells; expected {expected}."
            )

    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate CI detected in frozen {budget} subset.")
    if not set(ids).issubset(set(train_ids)):
        raise RuntimeError(f"Frozen {budget} subset contains CI outside train split.")

    return ids, payload


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.float()
    mask_f = mask.float()
    denom = mask_f.sum()
    if denom.item() <= 0:
        raise RuntimeError("Batch contains no valid measured grids.")
    return (((pred_f - target_f) ** 2) * mask_f).sum() / denom


class MetricAccumulator:
    def __init__(self):
        self.n = 0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.y_sum = 0.0
        self.y2_sum = 0.0
        self.macro_mae = []
        self.macro_rmse = []
        self.macro_r2 = []
        self.cells = 0

    def update(self, pred_gain_db, true_gain_db, mask):
        for i in range(pred_gain_db.shape[0]):
            valid = mask[i, 0] > 0.5
            if not torch.any(valid):
                continue

            pred = pred_gain_db[i, 0][valid].detach().double().cpu()
            true = true_gain_db[i, 0][valid].detach().double().cpu()
            err = pred - true

            n_i = int(true.numel())
            self.n += n_i
            self.abs_sum += float(err.abs().sum())
            self.sq_sum += float((err * err).sum())
            self.y_sum += float(true.sum())
            self.y2_sum += float((true * true).sum())
            self.cells += 1

            self.macro_mae.append(float(err.abs().mean()))
            self.macro_rmse.append(float(torch.sqrt((err * err).mean())))

            sse = float((err * err).sum())
            true_mean = float(true.mean())
            sst = float(((true - true_mean) ** 2).sum())
            if sst > 1e-12:
                self.macro_r2.append(1.0 - sse / sst)

    def result(self):
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


def make_loader(ds, batch_size, shuffle, seed, num_workers, device):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator if shuffle else None,
        persistent_workers=(num_workers > 0),
    )


def build_model(args):
    frequency_hz = args.target_frequency_ghz * 1e9
    k = 2.0 * math.pi * frequency_hz / 3e8
    return PhysicsInformedModel(
        k=k,
        dx=args.target_dx_m,
        lambda_d=args.lambda_data,
        lambda_phy=args.lambda_phy,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_num = 0.0
    loss_den = 0.0
    metrics = MetricAccumulator()

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        e_inc_raw = batch["einc_raw"].to(device, non_blocking=True)
        target = batch["target_norm"].to(device, non_blocking=True)
        true_gain = batch["gain_db"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        chi = x[:, 0:1] * 3.0

        out = model(
            x,
            e_inc_raw,
            chi,
            label=None,
            compute_physics_loss=False,
        )
        pred = out["PL_pred"].float()

        diff2 = ((pred - target.float()) ** 2) * mask.float()
        loss_num += float(diff2.sum())
        loss_den += float(mask.float().sum())

        pred_gain = pred * DB_SCALE + DB_MIN
        metrics.update(pred_gain, true_gain.float(), mask.float())

    result = metrics.result()
    result["masked_mse_norm"] = loss_num / max(loss_den, 1.0)
    return result


def write_csv_row(path: Path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    args = parse_args()

    if args.mode == "zero-shot":
        if args.budget != "0pct":
            raise ValueError("zero-shot requires --budget 0pct.")
        if args.source_checkpoint is None:
            raise ValueError("zero-shot requires --source-checkpoint.")
    elif args.mode == "finetune":
        if args.budget not in EXPECTED_SUBSET_COUNTS:
            raise ValueError("finetune budget must be 1pct/5pct/10pct/20pct.")
        if args.source_checkpoint is None:
            raise ValueError("finetune requires --source-checkpoint.")
    elif args.mode == "scratch" and args.budget == "0pct":
        raise ValueError("scratch cannot use 0pct.")

    seed_everything(args.seed)

    prepared_root = args.prepared_root.resolve()
    split_json = args.split_json.resolve()
    split_payload = json.loads(split_json.read_text(encoding="utf-8"))

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    carrier_hz = args.target_frequency_ghz * 1e9

    train_ds = None
    train_loader = None
    frozen_train_ids = []

    if args.mode != "zero-shot":
        frozen_train_ids, split_payload = get_frozen_train_ids(split_json, args.budget)
        train_ds = PEFNetRSRPSetDataset(
            prepared_root=prepared_root,
            split_json=split_json,
            split="train",
            carrier_frequency_hz=carrier_hz,
            dx_m=args.target_dx_m,
            cell_ids=frozen_train_ids,
            tx_row=args.tx_row,
            tx_col=args.tx_col,
        )

    val_ds = None
    if args.mode != "zero-shot":
        val_ds = PEFNetRSRPSetDataset(
            prepared_root=prepared_root,
            split_json=split_json,
            split="val",
            carrier_frequency_hz=carrier_hz,
            dx_m=args.target_dx_m,
            tx_row=args.tx_row,
            tx_col=args.tx_col,
        )

    test_ds = PEFNetRSRPSetDataset(
        prepared_root=prepared_root,
        split_json=split_json,
        split="test",
        carrier_frequency_hz=carrier_hz,
        dx_m=args.target_dx_m,
        tx_row=args.tx_row,
        tx_col=args.tx_col,
    )

    if train_ds is not None:
        train_set = set(train_ds.cell_ids)
        val_set = set(val_ds.cell_ids)
        test_set = set(test_ds.cell_ids)
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise RuntimeError("Data leakage detected: train/val/test CI overlap.")

        train_loader = make_loader(
            train_ds,
            args.batch_size,
            True,
            args.seed,
            args.num_workers,
            device,
        )

    val_loader = None
    if val_ds is not None:
        val_loader = make_loader(
            val_ds,
            args.batch_size,
            False,
            args.seed,
            args.num_workers,
            device,
        )

    test_loader = make_loader(
        test_ds,
        args.batch_size,
        False,
        args.seed,
        args.num_workers,
        device,
    )

    model = build_model(args)
    source_ckpt = None

    if args.mode in {"zero-shot", "finetune"}:
        source_ckpt = args.source_checkpoint.resolve()
        load_model_weights_strict(model, source_ckpt)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())

    probe_loader = train_loader if train_loader is not None else test_loader
    probe = next(iter(probe_loader))
    probe_x = probe["input"].to(device)
    probe_einc = probe["einc_raw"].to(device)
    probe_chi = probe_x[:, 0:1] * 3.0

    model.eval()
    with torch.no_grad():
        probe_out = model(
            probe_x,
            probe_einc,
            probe_chi,
            label=None,
            compute_physics_loss=False,
        )["PL_pred"]

    expected_shape = (
        probe_x.shape[0],
        1,
        probe_x.shape[-2],
        probe_x.shape[-1],
    )
    if tuple(probe_out.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape {tuple(probe_out.shape)}; "
            f"expected {expected_shape}."
        )

    run_dir = (
        args.output_root.resolve()
        / f"pefnet_{args.mode}_{args.budget}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if train_ds is not None:
        (run_dir / "train_cell_ids.txt").write_text(
            "\n".join(train_ds.cell_ids) + "\n",
            encoding="utf-8",
        )

    lr = args.lr
    if lr is None:
        lr = 4e-4 if args.mode == "finetune" else 4e-3

    config = {
        "model": "PEFNet",
        "mode": args.mode,
        "budget": args.budget,
        "seed": args.seed,
        "prepared_root": str(prepared_root),
        "split_json": str(split_json),
        "split_version": split_payload.get("version"),
        "split_seed": split_payload.get("split_seed"),
        "subset_seed": split_payload.get("subset_seed"),
        "frozen_train_cells": len(train_ds) if train_ds is not None else 0,
        "frozen_train_ids_sha256": (
            sha256_ids(train_ds.cell_ids) if train_ds is not None else None
        ),
        "val_cells": len(val_ds) if val_ds is not None else 0,
        "test_cells": len(test_ds),
        "source_checkpoint": str(source_ckpt) if source_ckpt else None,
        "source_checkpoint_sha256": sha256_file(source_ckpt),
        "model_parameters": total_params,
        "target_frequency_ghz": args.target_frequency_ghz,
        "target_dx_m": args.target_dx_m,
        "tx_grid_index": [args.tx_row, args.tx_col],
        "lambda_data": args.lambda_data,
        "lambda_phy": args.lambda_phy,
        "optimizer": "AdamW",
        "lr_initial": None if args.mode == "zero-shot" else lr,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "max_epochs": args.max_epochs,
        "eta_min": args.eta_min,
        "batch_size": args.batch_size,
        "val_interval_epochs": args.val_interval,
        "early_stopping_patience_epochs": args.patience_epochs,
        "grad_clip": args.grad_clip,
        "metric_mask": "measured strict-outdoor grids only",
        "db_normalization": {
            "min": DB_MIN,
            "max": DB_MAX,
            "clip": False,
        },
    }
    save_json(run_dir / "config.json", config)

    if args.mode == "zero-shot":
        t0 = time.time()
        test = evaluate(model, test_loader, device)
        test.update({
            "model": "PEFNet",
            "mode": "zero-shot",
            "budget": "0pct",
            "seed": args.seed,
            "train_cells": 0,
            "test_elapsed_sec": time.time() - t0,
        })
        save_json(run_dir / "test_metrics.json", test)
        print(json.dumps(test, indent=2, ensure_ascii=False))
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=args.eta_min,
    )

    initial_val = evaluate(model, val_loader, device)
    initial_val["epoch"] = 0
    save_json(run_dir / "initial_val_metrics.json", initial_val)

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
        train_num = 0.0
        train_den = 0.0
        train_phy_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            x = batch["input"].to(device, non_blocking=True)
            e_inc_raw = batch["einc_raw"].to(device, non_blocking=True)
            target = batch["target_norm"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            chi = x[:, 0:1] * 3.0

            optimizer.zero_grad(set_to_none=True)

            out = model(
                x,
                e_inc_raw,
                chi,
                label=None,
                compute_physics_loss=True,
            )
            pred = out["PL_pred"]
            loss_data = masked_mse(pred, target, mask)
            loss_phy = out["loss_phy"].float()
            loss = args.lambda_data * loss_data + args.lambda_phy * loss_phy

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}: "
                    f"total={float(loss.detach().cpu())}, "
                    f"data={float(loss_data.detach().cpu())}, "
                    f"phy={float(loss_phy.detach().cpu())}"
                )

            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                pred_f = pred.float()
                target_f = target.float()
                mask_f = mask.float()
                diff2 = ((pred_f - target_f) ** 2) * mask_f
                train_num += float(diff2.sum())
                train_den += float(mask_f.sum())
                train_phy_sum += float(loss_phy.detach().cpu())
                train_batches += 1

        train_mse = train_num / max(train_den, 1.0)
        train_phy = train_phy_sum / max(train_batches, 1)

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
            "train_physics_loss": train_phy,
            "lr": current_lr,
            "validated": int(do_val),
            "val_masked_mse_norm": "",
            "val_mae_db": "",
            "val_rmse_db": "",
            "val_r2": "",
        }

        if do_val:
            val = evaluate(model, val_loader, device)
            row.update({
                "val_masked_mse_norm": val["masked_mse_norm"],
                "val_mae_db": val["mae_db"],
                "val_rmse_db": val["rmse_db"],
                "val_r2": val["r2"],
            })

            if val["masked_mse_norm"] < best_val - args.min_delta:
                best_val = val["masked_mse_norm"]
                best_epoch = epoch
                last_improve_epoch = epoch
                torch.save(model.state_dict(), best_path)
                save_json(
                    run_dir / "best_val_metrics.json",
                    {"epoch": epoch, **val},
                )

        write_csv_row(history_path, row)

        if epoch - last_improve_epoch >= args.patience_epochs:
            break

    load_model_weights_strict(model, best_path)
    model = model.to(device)

    t_test = time.time()
    test = evaluate(model, test_loader, device)
    test.update({
        "model": "PEFNet",
        "mode": args.mode,
        "budget": args.budget,
        "seed": args.seed,
        "train_cells": len(train_ds),
        "frozen_train_ids_sha256": sha256_ids(train_ds.cell_ids),
        "best_epoch": best_epoch,
        "best_val_masked_mse_norm": best_val,
        "learning_rate_initial": lr,
        "test_elapsed_sec": time.time() - t_test,
        "total_elapsed_sec": time.time() - total_start,
    })
    save_json(run_dir / "test_metrics.json", test)
    print(json.dumps(test, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
