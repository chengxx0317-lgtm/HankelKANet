import os,  time
import json
import math
import random
import numpy as np
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from loaders import Dataset_RadioMapSeer          
from modules import HKANNet
from checkpointer import Checkpointer
import torch
import matplotlib
from torch.optim.lr_scheduler import CosineAnnealingLR
matplotlib.use("Agg")   # 使用无 GUI 后端，生成图片但不弹窗


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def main():
    device = 'cuda'

    # ===================== 正式实验固定参数 =====================
    RUN_SEED = 0
    Z_MIN = math.pi * 1e-3
    KAPPA_MIN = 0.1
    KAPPA_MAX = 10.0
    KAPPA_INIT = math.pi

    # 固定随机种子，后续做 mean ± std 多次独立实验
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RUN_SEED)
        torch.cuda.manual_seed_all(RUN_SEED)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False

    # 捕获 Ctrl+C 并保存 checkpoint
    import signal
    import sys

    ckpt_path = f"checkpoint_hkan_20seed{RUN_SEED}.pt"
    ckpt = None

    def save_on_interrupt(sig, frame):
        print("\n[Interrupt] Caught Ctrl+C, saving checkpoint before exit...")
        try:
            if ckpt is not None:
                ckpt.save(epoch, step, best_val, model, optimizer, scheduler, scaler)
                print("Checkpoint saved. Exiting safely.")
            else:
                print("Checkpointer not initialized, exiting without saving checkpoint.")
        except NameError:
            print("Training variables not ready, exiting without saving checkpoint.")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_on_interrupt)

    max_epochs  = 250
    batch_size = 32
    # Early stopping
    early_stop_patience = 50
    early_stop_min_delta = 1e-6
    print_interval = 1000
    vis_interval = 1000
    save_dir = f"./train_visions_hkan_20seed{RUN_SEED}"
    os.makedirs(save_dir, exist_ok=True)
    gc_root = Path(save_dir) / "gc_artifacts"
    gc_snap_dir = gc_root / "snapshots"
    gc_snap_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = f"./models/hkan_20seed{RUN_SEED}.pt"
    train_log_path = os.path.join(save_dir, f"train_log_seed{RUN_SEED}.txt")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    DEBUG = dict(
        grad_clip=1.0,
        log_lr=True,
        log_time=True,
    )

    def visualize_batch(inputs, label, pred, step, phase='train'):
        vmin = float(label.min().cpu())
        vmax = float(label.max().cpu())

        i = 0
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))

        axs[0].set_title('Label')
        im0 = axs[0].imshow(label[i, 0].detach().cpu().numpy(),
                            cmap='viridis', vmin=vmin, vmax=vmax)
        axs[0].axis('off')
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        pred_i = pred[i, 0].detach().cpu()
        label_i = label[i, 0].detach().cpu()

        mse = torch.mean((pred_i - label_i) ** 2).item()
        mae = torch.mean(torch.abs(pred_i - label_i)).item()

        axs[1].set_title(f'Prediction\nMSE={mse:.4f}, MAE={mae:.4f}')
        im1 = axs[1].imshow(pred_i.numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axs[1].axis('off')
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{phase}_vis_step_{step:06d}.png"))
        plt.close(fig)

    # ===================== 数据集 =====================
    train_set = Dataset_RadioMapSeer(phase='train', img_size=256)
    val_set = Dataset_RadioMapSeer(phase='val', img_size=256)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(RUN_SEED)



    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=seed_worker
    )

    model = HKANNet(
        base_ch=32,
        phys_ch=32,
        learnable_k=True,
        z_min=Z_MIN,
        kappa_min=KAPPA_MIN,
        kappa_max=KAPPA_MAX,
        kappa_init=KAPPA_INIT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Run seed   : {RUN_SEED}")
    print(f"z_min      : {Z_MIN:.10f}")
    print(f"kappa range: [{KAPPA_MIN}, {KAPPA_MAX}]")
    print(f"kappa init : {KAPPA_INIT:.10f}")
    print(f"Parameters : {total_params:,} ({total_params / 1e6:.6f} M)")
    GC_N = 50
    GC_SEED = 2025

    # 指定保存模型快照的训练轮次
    GC_SAVE_EPOCHS = {
        1, 10, 20, 30, 40, 50,
        60, 70, 80, 90, 100
    }

    idx_path = gc_root / f"eval_indices_val{GC_N}.npy"

    if idx_path.exists():
        gc_indices = np.load(idx_path)
    else:
        rng = np.random.default_rng(GC_SEED)
        gc_indices = rng.choice(len(val_set), size=GC_N, replace=False)
        gc_indices = np.sort(gc_indices)
        np.save(idx_path, gc_indices)

    # 记录元信息，方便之后离线复现
    meta_path = gc_root / "meta.json"

    if not meta_path.exists():
        meta = {
            "gc_n": int(GC_N),
            "gc_seed": int(GC_SEED),
            "val_len": int(len(val_set)),
            "dataset_dir": getattr(val_set, "dir_dataset", None),
            "img_size": getattr(val_set, "img_size", 256),
            "numTx": getattr(val_set, "numTx", None),
            "model_name": model.__class__.__name__,
            "model_kwargs": {
                "base_ch": 32,
                "phys_ch": 32,
                "orders": [0, 1],
                "learnable_k": True,
                "z_min": Z_MIN,
                "kappa_min": KAPPA_MIN,
                "kappa_max": KAPPA_MAX,
                "kappa_init": KAPPA_INIT,
            },
            "run_seed": RUN_SEED,
            "input_channels_hint": 2,  # Datasetstack了2通道：bld+r_map
            "db_min": -147.0,
            "db_max": -47.84
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max= max_epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # ===================== Checkpoint 恢复 =====================
    ckpt = Checkpointer(ckpt_path)
    ckpt.load_if_exists(model, optimizer, scheduler, scaler, device)
    start_epoch = ckpt.start_epoch                                     
    step = ckpt.step
    best_val = ckpt.best_val
    bad_epochs = 0
    # ===================== 训练 & 验证循环 =====================
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0_epoch = time.time()

        for it, (inputs, label,sample_id) in enumerate(train_loader, 1):

            inputs = inputs.to(device, non_blocking=True)

            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=False):
                pred = model(inputs)
                data_loss = criterion(pred, label)
                loss = data_loss

            scaler.scale(loss).backward()
            if DEBUG["grad_clip"] is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), DEBUG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            step += 1
            epoch_loss += loss.item() * inputs.size(0)

            if step % 5000 == 0:
                ckpt.save(epoch, step, best_val, model, optimizer, scheduler, scaler)
                #print(f"[Checkpoint] Saved at step {step}")

            if step % print_interval == 0:
                msg = f"[Epoch {epoch:02d} | Step {step:05d}] loss={loss.item():.6f}"
                if DEBUG["log_lr"]:
                    msg += f" | lr={optimizer.param_groups[0]['lr']:.3e}"
                if DEBUG["log_time"]:
                    dt = time.time() - t0_epoch
                    msg += f" | dt_batch={dt:.1f}s"
                print(msg)
                with open(train_log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

            '''if step % vis_interval == 0:
                visualize_batch(inputs, label, pred, step, phase='train')'''

        # ========== 验证阶段 with 反归一化(dB)指标 ==========
        model.eval()

        val_loss = 0.0
        val_mse = 0.0
        val_mae = 0.0
        val_rmse_db = 0.0
        val_mae_db = 0.0
        val_r2 = 0.0
        count = 0

        DB_MAX = -47.84
        DB_MIN = -147.0
        scale = DB_MAX - DB_MIN

        with torch.no_grad():
            for i, (inputs, label, sample_id) in enumerate(val_loader):
                inputs = inputs.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                pred = model(inputs)

                mse = torch.mean((pred - label) ** 2, dim=[1, 2, 3])
                mae = torch.mean(torch.abs(pred - label), dim=[1, 2, 3])

                pred_db = pred * scale + DB_MIN
                label_db = label * scale + DB_MIN

                rmse_db = torch.sqrt(torch.mean((pred_db - label_db) ** 2, dim=[1, 2, 3]))
                mae_db = torch.mean(torch.abs(pred_db - label_db), dim=[1, 2, 3])
                # 每张路径损耗图分别计算 R²
                flat_pred_db = pred_db.flatten(start_dim=1)
                flat_label_db = label_db.flatten(start_dim=1)

                ss_res = torch.sum(
                    (flat_label_db - flat_pred_db) ** 2,
                    dim=1
                )

                label_mean = torch.mean(
                    flat_label_db,
                    dim=1,
                    keepdim=True
                )

                ss_tot = torch.sum(
                    (flat_label_db - label_mean) ** 2,
                    dim=1
                )
                # 防止极端情况下标签图完全恒定，导致除零
                r2 = torch.where(
                    ss_tot > 1e-12,
                    1.0 - ss_res / ss_tot.clamp_min(1e-12),
                    torch.zeros_like(ss_res)
                )

                val_loss += mse.sum().item()
                val_mse += mse.sum().item()
                val_mae += mae.sum().item()
                val_rmse_db += rmse_db.sum().item()
                val_mae_db += mae_db.sum().item()
                val_r2 += r2.sum().item()
                count += inputs.size(0)



        val_loss /= count
        val_mse /= count
        val_mae /= count
        val_rmse_db /= count
        val_mae_db /= count
        val_r2 /= count
        val_msg = (f"Epoch {epoch:02d} | "
              f"ValLoss={val_loss:.6f}, "
              f"MSE={val_mse:.6f}, MAE={val_mae:.6f}, "
              f"MAE(dB)={val_mae_db:.3f},RMSE(dB)={val_rmse_db:.3f},"
              f"R2={val_r2:.5f}"
                   )
        print(val_msg)
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(val_msg + "\n")
        # ===================== Best model and early stopping =====================
        improved = val_loss < (best_val - early_stop_min_delta)

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
                scaler
            )

            torch.save(
                model.state_dict(),
                model_save_path
            )

            status_msg = (
                f"--> Saved new best model at epoch {epoch:03d} | "
                f"ValLoss={best_val:.6f}"
            )

        else:
            bad_epochs += 1

            status_msg = (
                f"--> No validation improvement: "
                f"{bad_epochs}/{early_stop_patience} | "
                f"Current ValLoss={val_loss:.6f} | "
                f"Best ValLoss={best_val:.6f}"
            )

        print(status_msg)

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(status_msg + "\n")
        # ===================== 保存GC指定轮次快照 =====================
        if epoch in GC_SAVE_EPOCHS:
            snapshot_path = gc_snap_dir / f"epoch_{epoch:04d}.pt"

            gc_pack = {
                "epoch": int(epoch),
                "step": int(step),
                "val_loss": float(val_loss),
                "val_mse": float(val_mse),
                "val_mae": float(val_mae),
                "val_rmse_db": float(val_rmse_db),
                "val_mae_db": float(val_mae_db),
                "val_r2": float(val_r2),
                "best_val": float(best_val),
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),

                # 将模型参数转移到CPU后保存，避免快照绑定GPU设备
                "model": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
            }

            torch.save(gc_pack, snapshot_path)

            gc_msg = (
                f"--> Saved GC snapshot: "
                f"{snapshot_path} | "
                f"Epoch={epoch:03d} | "
                f"ValLoss={val_loss:.6f}"
            )

            print(gc_msg)

            with open(train_log_path, "a", encoding="utf-8") as f:
                f.write(gc_msg + "\n")

        # ===================== Early stopping =====================
        if bad_epochs >= early_stop_patience:
            stop_msg = (
                f"[Early stopping] Validation loss did not improve "
                f"for {early_stop_patience} consecutive epochs. "
                f"Best ValLoss={best_val:.6f}"
            )

            print(stop_msg)

            with open(train_log_path, "a", encoding="utf-8") as f:
                f.write(stop_msg + "\n")

            break

        # ===================== LR scheduler =====================
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        lr_msg = f"--> Learning rate updated to {current_lr:.6e}"

        print(lr_msg)

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(lr_msg + "\n")




if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()