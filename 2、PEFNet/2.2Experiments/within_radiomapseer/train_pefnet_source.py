from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from pefnet_model import PhysicsInformedModel
from pefnet_radiomapseer_dataset import DatasetRadioMapSeer


DB_MIN = -147.0
DB_MAX = -47.84
DB_SCALE = DB_MAX - DB_MIN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-root", type=Path, default=Path("./pefnet_source_runs"))
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-frequency-ghz", type=float, default=5.9)
    p.add_argument("--source-dx-m", type=float, default=1.0)
    p.add_argument("--lambda-data", type=float, default=1.0)
    p.add_argument("--lambda-phy", type=float, default=1.0)
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


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mse_sum = 0.0
    mae_db_sum = 0.0
    rmse_db_sum = 0.0
    count = 0

    for inputs, label, e_inc_raw, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        e_inc_raw = e_inc_raw.to(device, non_blocking=True)
        chi = inputs[:, 0:1] * 3.0

        out = model(
            inputs,
            e_inc_raw,
            chi,
            label=None,
            compute_physics_loss=False,
        )
        pred = out["PL_pred"].float()
        label_f = label.float()
        bs = inputs.shape[0]
        count += bs

        mse = torch.mean((pred - label_f) ** 2, dim=[1, 2, 3])
        mse_sum += float(mse.sum())

        pred_db = pred * DB_SCALE + DB_MIN
        label_db = label_f * DB_SCALE + DB_MIN
        mae_db = torch.mean(torch.abs(pred_db - label_db), dim=[1, 2, 3])
        rmse_db = torch.sqrt(torch.mean((pred_db - label_db) ** 2, dim=[1, 2, 3]))

        mae_db_sum += float(mae_db.sum())
        rmse_db_sum += float(rmse_db.sum())

    return {
        "val_mse": mse_sum / max(count, 1),
        "val_mae_db": mae_db_sum / max(count, 1),
        "val_rmse_db": rmse_db_sum / max(count, 1),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    source_frequency_hz = args.source_frequency_ghz * 1e9

    train_set = DatasetRadioMapSeer(
        phase="train",
        dir_dataset=args.dataset_root,
        img_size=256,
        f=source_frequency_hz,
        dx=args.source_dx_m,
    )
    val_set = DatasetRadioMapSeer(
        phase="val",
        dir_dataset=args.dataset_root,
        img_size=256,
        f=source_frequency_hz,
        dx=args.source_dx_m,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    k = 2.0 * math.pi * source_frequency_hz / 3e8
    model = PhysicsInformedModel(
        k=k,
        dx=args.source_dx_m,
        lambda_d=args.lambda_data,
        lambda_phy=args.lambda_phy,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(0.5 * args.epochs)),
        eta_min=1e-6,
    )

    run_dir = args.output_root.resolve() / f"pefnet_source_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pt"

    best_val = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_sum = 0.0
        data_sum = 0.0
        phy_sum = 0.0
        batches = 0

        for inputs, label, e_inc_raw, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            e_inc_raw = e_inc_raw.to(device, non_blocking=True)
            chi = inputs[:, 0:1] * 3.0

            out = model(
                inputs,
                e_inc_raw,
                chi,
                label=label,
                compute_physics_loss=True,
            )
            loss = out["loss_total"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite source loss at epoch {epoch}: "
                    f"{float(loss.detach().cpu())}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_sum += float(loss.detach().cpu())
            data_sum += float(out["loss_data"].detach().cpu())
            phy_sum += float(out["loss_phy"].detach().cpu())
            batches += 1

        val = validate(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_total": total_sum / max(batches, 1),
            "train_data": data_sum / max(batches, 1),
            "train_phy": phy_sum / max(batches, 1),
            **val,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Seed {args.seed} | Epoch {epoch:03d} | "
            f"ValMSE={val['val_mse']:.6f} | "
            f"MAE={val['val_mae_db']:.4f} dB | "
            f"RMSE={val['val_rmse_db']:.4f} dB"
        )

        if val["val_mse"] < best_val:
            best_val = val["val_mse"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.resolve())
    config.update({
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "checkpoint": str(best_path),
    })
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
