import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from checkpointer import Checkpointer
from loaders_angular import Dataset_RadioMapSeer_Angular
from modules_angular import HKANAngularNet


ANGULAR_MAX_ORDER = 2
# Run the three required experiments separately:
#   1  -> complete angular modes |n| <= 1
#   2  -> complete angular modes |n| <= 2
#   3  -> complete angular modes |n| <= 3

RUN_SEED = 0
DATASET_DIR = "your_dataset_path"

MAX_EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 4e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 50
EARLY_STOP_MIN_DELTA = 1e-6

PRINT_INTERVAL = 500
CHECKPOINT_INTERVAL = 5000
NUM_WORKERS = 0
IMG_SIZE = 256

# Finalized Hankel settings: identical to the radial baseline and order ablation
Z_MIN = math.pi * 1e-3
KAPPA_MIN = 0.1
KAPPA_MAX = 10.0
KAPPA_INIT = math.pi

DB_MIN = -147.0
DB_MAX = -47.84

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Keep the same gradient clipping behavior as the current baseline training code.
GRAD_CLIP = 1.0


# ============================================================================
# Utilities
# ============================================================================
def validate_max_order(max_order):
    max_order = int(max_order)

    if max_order < 0:
        raise ValueError("ANGULAR_MAX_ORDER must be >= 0.")

    return max_order


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # PyTorch default is benchmark=False; keep it explicit for reproducibility.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate(model, loader, device):

    model.eval()

    scale = DB_MAX - DB_MIN

    total_mse = 0.0
    total_mae = 0.0
    total_rmse_db = 0.0
    total_mae_db = 0.0
    total_r2 = 0.0
    count = 0

    with torch.no_grad():
        for inputs, label, _sample_id in loader:
            inputs = inputs.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            pred = model(inputs)

            mse = torch.mean(
                (pred - label) ** 2,
                dim=[1, 2, 3],
            )
            mae = torch.mean(
                torch.abs(pred - label),
                dim=[1, 2, 3],
            )

            pred_db = pred * scale + DB_MIN
            label_db = label * scale + DB_MIN

            rmse_db = torch.sqrt(
                torch.mean(
                    (pred_db - label_db) ** 2,
                    dim=[1, 2, 3],
                )
            )

            mae_db = torch.mean(
                torch.abs(pred_db - label_db),
                dim=[1, 2, 3],
            )

            flat_pred_db = pred_db.flatten(start_dim=1)
            flat_label_db = label_db.flatten(start_dim=1)

            ss_res = torch.sum(
                (flat_label_db - flat_pred_db) ** 2,
                dim=1,
            )

            label_mean = torch.mean(
                flat_label_db,
                dim=1,
                keepdim=True,
            )

            ss_tot = torch.sum(
                (flat_label_db - label_mean) ** 2,
                dim=1,
            )

            r2 = torch.where(
                ss_tot > 1e-12,
                1.0 - ss_res / ss_tot.clamp_min(1e-12),
                torch.zeros_like(ss_res),
            )

            total_mse += mse.sum().item()
            total_mae += mae.sum().item()
            total_rmse_db += rmse_db.sum().item()
            total_mae_db += mae_db.sum().item()
            total_r2 += r2.sum().item()

            count += inputs.size(0)

    return {
        "mse_norm": total_mse / count,
        "mae_norm": total_mae / count,
        "mae_db": total_mae_db / count,
        "rmse_db": total_rmse_db / count,
        "r2": total_r2 / count,
        "n_samples": int(count),
    }


def main():
    max_order = validate_max_order(ANGULAR_MAX_ORDER)
    set_seed(RUN_SEED)

    exp_name = f"angular_response_leq{max_order}_seed{RUN_SEED}"

    save_root = Path("./angular_ablation_results") / exp_name
    save_root.mkdir(parents=True, exist_ok=True)

    model_dir = Path("./models_angular_ablation")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = model_dir / f"{exp_name}.pt"
    ckpt_path = save_root / f"checkpoint_{exp_name}.pt"
    train_log_path = save_root / "train_log.txt"
    config_path = save_root / "config.json"
    test_result_path = save_root / "test_metrics.json"

    config = {
        "encoding": "complete_tx_centered_angular_hankel",
        "angular_max_order": max_order,
        "angular_notation": f"|n| <= {max_order}",
        "represented_nonnegative_orders": list(range(max_order + 1)),
        "angular_components_per_order": ["real", "imag"],
        "angular_component_transform": "signed_log1p",
        "run_seed": RUN_SEED,
        "dataset_dir": DATASET_DIR,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "grad_clip": GRAD_CLIP,
        "z_min": Z_MIN,
        "kappa_min": KAPPA_MIN,
        "kappa_max": KAPPA_MAX,
        "kappa_init": KAPPA_INIT,
        "db_min": DB_MIN,
        "db_max": DB_MAX,
        "device": DEVICE,
        "input_channels": [
            "building_map",
            "baseline_normalized_r_map",
            "tx_centered_phi_map",
        ],
        "notes": (
            "Backbone/training/data split are aligned with the finalized radial HKANNet. "
            "The physical encoding is replaced by complete angular Hankel modes."
        ),
    }

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 84)
    print("Complete Tx-centered angular Hankel experiment")
    print(f"Angular order : |n| <= {max_order}")
    print(f"Run seed      : {RUN_SEED}")
    print(f"Device        : {DEVICE}")
    print(f"Experiment    : {exp_name}")
    print(f"Output dir    : {save_root}")
    print(f"z_min         : {Z_MIN:.10f}")
    print(f"kappa range   : [{KAPPA_MIN}, {KAPPA_MAX}]")
    print(f"kappa init    : {KAPPA_INIT:.10f}")
    print("=" * 84)


    train_set = Dataset_RadioMapSeer_Angular(
        phase="train",
        dir_dataset=DATASET_DIR,
        img_size=IMG_SIZE,
    )
    val_set = Dataset_RadioMapSeer_Angular(
        phase="val",
        dir_dataset=DATASET_DIR,
        img_size=IMG_SIZE,
    )
    test_set = Dataset_RadioMapSeer_Angular(
        phase="test",
        dir_dataset=DATASET_DIR,
        img_size=IMG_SIZE,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(RUN_SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------
    model = HKANAngularNet(
        base_ch=32,
        phys_ch=32,
        max_order=max_order,
        learnable_k=True,
        z_min=Z_MIN,
        kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX,
        kappa_init=KAPPA_INIT,
    ).to(DEVICE)

    n_params = count_parameters(model)

    print(
        f"Trainable/model parameters: "
        f"{n_params:,} ({n_params / 1e6:.4f} M)"
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(False)
    )

    # ------------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------------
    ckpt = Checkpointer(str(ckpt_path))

    ckpt.load_if_exists(
        model,
        optimizer,
        scheduler,
        scaler,
        DEVICE,
    )

    start_epoch = ckpt.start_epoch
    step = ckpt.step
    best_val = ckpt.best_val
    bad_epochs = 0
    current_epoch = start_epoch

    def save_on_interrupt(sig, frame):
        print(
            "\n[Interrupt] Caught Ctrl+C, "
            "saving checkpoint before exit..."
        )

        try:
            ckpt.save(
                current_epoch,
                step,
                best_val,
                model,
                optimizer,
                scheduler,
                scaler,
            )
            print("Checkpoint saved. Exiting safely.")
        except Exception as exc:
            print(f"Checkpoint save failed: {exc}")

        sys.exit(0)

    signal.signal(signal.SIGINT, save_on_interrupt)

    # ------------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------------
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        current_epoch = epoch

        model.train()
        epoch_loss = 0.0
        t0_epoch = time.time()

        for _it, (inputs, label, sample_id) in enumerate(
            train_loader,
            1,
        ):
            inputs = inputs.to(
                DEVICE,
                non_blocking=True,
            )
            label = label.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=(False)
            ):
                pred = model(inputs)
                data_loss = criterion(pred, label)
                loss = data_loss

            scaler.scale(loss).backward()

            if GRAD_CLIP is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

            scaler.step(optimizer)
            scaler.update()

            step += 1
            epoch_loss += loss.item() * inputs.size(0)

            if step % CHECKPOINT_INTERVAL == 0:
                ckpt.save(
                    epoch,
                    step,
                    best_val,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                )

            if step % PRINT_INTERVAL == 0:
                elapsed = time.time() - t0_epoch

                msg = (
                    f"[Epoch {epoch:03d} | Step {step:06d}] "
                    f"loss={loss.item():.6f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} | "
                    f"dt_epoch={elapsed:.1f}s"
                )

                print(msg)

                with open(
                    train_log_path,
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(msg + "\n")

        # --------------------------------------------------------------------
        # Validation: same per-map metric protocol as the baseline.
        # --------------------------------------------------------------------
        val_metrics = evaluate(
            model,
            val_loader,
            DEVICE,
        )

        val_loss = val_metrics["mse_norm"]

        val_msg = (
            f"Epoch {epoch:03d} | "
            f"ValLoss={val_loss:.6f}, "
            f"MSE={val_metrics['mse_norm']:.6f}, "
            f"MAE={val_metrics['mae_norm']:.6f}, "
            f"MAE(dB)={val_metrics['mae_db']:.3f}, "
            f"RMSE(dB)={val_metrics['rmse_db']:.3f}, "
            f"R2={val_metrics['r2']:.5f}"
        )

        print(val_msg)

        with open(
            train_log_path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(val_msg + "\n")

        # --------------------------------------------------------------------
        # Best model + early stopping
        # --------------------------------------------------------------------
        improved = val_loss < (
            best_val - EARLY_STOP_MIN_DELTA
        )

        if improved:
            best_val = val_loss
            bad_epochs = 0

            ckpt.save(
                epoch,
                step,
                best_val,
                model,
                optimizer,
                scheduler,
                scaler,
            )

            torch.save(
                model.state_dict(),
                model_save_path,
            )

            status_msg = (
                f"--> Saved new best model at epoch {epoch:03d} | "
                f"ValLoss={best_val:.6f}"
            )

        else:
            bad_epochs += 1

            status_msg = (
                "--> No validation improvement: "
                f"{bad_epochs}/{EARLY_STOP_PATIENCE} | "
                f"Current ValLoss={val_loss:.6f} | "
                f"Best ValLoss={best_val:.6f}"
            )

        print(status_msg)

        with open(
            train_log_path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(status_msg + "\n")

        if bad_epochs >= EARLY_STOP_PATIENCE:
            stop_msg = (
                "[Early stopping] Validation loss did not improve "
                f"for {EARLY_STOP_PATIENCE} consecutive epochs. "
                f"Best ValLoss={best_val:.6f}"
            )

            print(stop_msg)

            with open(
                train_log_path,
                "a",
                encoding="utf-8",
            ) as f:
                f.write(stop_msg + "\n")

            break

        scheduler.step()

        lr_msg = (
            "--> Learning rate updated to "
            f"{optimizer.param_groups[0]['lr']:.6e}"
        )

        print(lr_msg)

        with open(
            train_log_path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(lr_msg + "\n")

    # ------------------------------------------------------------------------
    # Test the validation-selected best checkpoint.
    # ------------------------------------------------------------------------
    if not model_save_path.exists():
        raise FileNotFoundError(
            f"No best model was saved at {model_save_path}"
        )

    print("\nLoading validation-selected best model for test evaluation...")

    state_dict = torch.load(
        model_save_path,
        map_location=DEVICE,
    )

    model.load_state_dict(state_dict, strict=True)

    test_metrics = evaluate(
        model,
        test_loader,
        DEVICE,
    )

    test_result = {
        "experiment": exp_name,
        "encoding": "complete_tx_centered_angular_hankel",
        "angular_max_order": max_order,
        "angular_notation": f"|n| <= {max_order}",
        "run_seed": RUN_SEED,
        "parameters": int(n_params),
        "parameters_million": n_params / 1e6,
        **test_metrics,
    }

    test_result_path.write_text(
        json.dumps(
            test_result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 84)
    print("TEST RESULT")
    print(f"Encoding : complete angular |n| <= {max_order}")
    print(f"R2       : {test_metrics['r2']:.6f}")
    print(f"MAE(dB)  : {test_metrics['mae_db']:.6f}")
    print(f"RMSE(dB) : {test_metrics['rmse_db']:.6f}")
    print(f"Params   : {n_params:,}")
    print(f"Saved to : {test_result_path}")
    print("=" * 84)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()

    try:
        mp.set_start_method(
            "spawn",
            force=True,
        )
    except RuntimeError:
        pass

    main()