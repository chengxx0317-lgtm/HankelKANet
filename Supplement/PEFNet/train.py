from torch.utils.data import DataLoader
from lib.modules import PhysicsInformedModel
from lib.loaders import Dataset_RadioMapSeer
import math
import os, torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import json
import numpy as np
from pathlib import Path
from checkpointer import Checkpointer
def main():
    device = 'cuda'
    epochs = 250
    batch_size = 16
    print_interval = 1000
    vis_interval = 1000
    save_dir = "./train_visions"
    os.makedirs(save_dir, exist_ok=True)

    # 模型保存路径
    model_save_path = r"D:\PycharmProjects\PythonProjects"
    # ====== 断点保存与日志路径======
    ckpt_path = os.path.join(save_dir, "checkpoint_PEFNet.pt")
    train_log_path = os.path.join(save_dir, "train_log.txt")

    # ======  GC 离线计算素材保存目录 ======
    gc_root = Path(save_dir) / "gc_artifacts"
    gc_snap_dir = gc_root / "snapshots"
    gc_snap_dir.mkdir(parents=True, exist_ok=True)


    train_set = Dataset_RadioMapSeer(phase='train', img_size=256)
    val_set   = Dataset_RadioMapSeer(phase='val', img_size=256)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True)
    val_loader   = DataLoader( val_set, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)


    # 模型与优化器

    k = 2 * math.pi * 5.9e9 / 3e8  # 波数
    dx =1
    model = PhysicsInformedModel(k, dx, lambda_d=1,).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    # ====== NEW: GC 固定32个验证样本索引 ======
    GC_N = 50
    GC_SEED = 2025
    idx_path = gc_root / f"eval_indices_val{GC_N}.npy"

    if idx_path.exists():
        gc_indices = np.load(idx_path)
    else:
        rng = np.random.default_rng(GC_SEED)
        gc_indices = rng.choice(len(val_set), size=GC_N, replace=False)
        gc_indices = np.sort(gc_indices)
        np.save(idx_path, gc_indices)

    # ======  GC meta ======
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
            "model_kwargs": {"k": float(k), "dx": float(dx), "lambda_d": 1.0},
            "input_channels_hint": 4,
            "db_min": -147.0,
            "db_max": -47.84
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


    # ======  Checkpointer 断点恢复======

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    ckpt = Checkpointer(ckpt_path)
    ckpt.load_if_exists(model, optimizer, scheduler, scaler, device)
    start_epoch = ckpt.start_epoch
    step = ckpt.step
    best_val_mse = ckpt.best_val

    # 可视化函数
    def visualize_batch(inputs, label, PL_init, PL_pred, step, phase='train'):
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import torch

        indices = np.random.choice(inputs.shape[0], size=1, replace=False)  # 随机选择样本

        # 统一色轴（按标签范围，保证对比一致性）
        vmin = float(label.min().cpu())
        vmax = float(label.max().cpu())

        for i in indices:
            # 保留 3 张图：Label、PL_init、PL_pred
            fig, axs = plt.subplots(1, 3, figsize=(12, 3))

            # 1 真实值（Label）
            axs[0].set_title('Label')
            im0 = axs[0].imshow(label[i, 0].detach().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
            axs[0].axis('off')
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            # 2 初始预测（PL_init）
            axs[1].set_title('PL_init')
            im1 = axs[1].imshow(PL_init[i, 0].detach().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
            axs[1].axis('off')
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            # 3 最终预测+误差指标（PL_pred）
            mse = torch.mean((PL_pred[i, 0] - label[i, 0]) ** 2).item()
            mae = torch.mean(torch.abs(PL_pred[i, 0] - label[i, 0])).item()
            axs[2].set_title(f'PL_pred\n(MSE={mse:.4f}, MAE={mae:.4f})')
            im2 = axs[2].imshow(PL_pred[i, 0].detach().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
            axs[2].axis('off')
            fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

            plt.tight_layout()

            # 保存路径
            plt.savefig(os.path.join(save_dir, f"{phase}_vis_step_{step:06d}_idx{i}.png"))
            plt.close(fig)


    # 训练循环

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        for inputs, label , E_inc_raw in train_loader:
            inputs, label ,E_inc_raw = inputs.to(device), label.to(device), E_inc_raw.to(device)
            #chi = TF.gaussian_blur(inputs[:, 0:1, :, :], kernel_size=(5, 5), sigma=(1.0, 1.0)) * 3.0
            chi = inputs[:, 0:1, :, :] * 3.0
            E_inc = inputs[:, 2:4, :, :]# 归一化的 E_inc 用于模型输入
            #chi = F.interpolate(chi, size=(inputs.shape[-2], inputs.shape[-1]), mode='nearest')

            out = model(inputs, E_inc_raw, chi, label)  # 使用未归一化的 E_inc_raw
            loss_total = out["loss_total"]

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            step += 1
            # ====== 每5000 step 保存一次断点 ======

            if step % 5000 == 0:
                ckpt.save(epoch, step, best_val_mse, model, optimizer, scheduler, scaler)

            # 打印损失
            if step % print_interval == 0:
                #print(f"\n【Chi Matrix Shape Check】")
                #print(f"chi矩阵尺寸: {chi.shape}")  # 应为 (batch_size, 1, height, width)
                msg = (f"[Epoch {epoch:02d} | Step {step:05d}] "
                      f"L_phy={out['loss_phy']:.6f} "
                      f"L_data={out['loss_data']:.6f} "
                      f"L_total={out['loss_total']:.6f}")
                print(msg)
                with open(train_log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")


            # 训练过程可视化
            '''if step % vis_interval == 0:
                visualize_batch(inputs, label, out["PL_init"], out["PL_pred"], step,phase='train')'''

        # ======================== 验证阶段 ========================
        model.eval()

        # --- 纯预测误差---
        val_mse = 0.0
        val_rmse = 0.0
        val_mae = 0.0

        # --- dB 指标 ---
        val_rmse_db = 0.0
        val_mae_db = 0.0

        # --- 训练目标监控（含物理项）---
        val_loss_total = 0.0
        val_loss_data = 0.0
        val_loss_phy = 0.0

        count = 0

        # RadioMapSeer 反归一化参数
        M1 = -47.84
        PL_trnc = -147.0
        scale = (M1 - PL_trnc)  # 99.16 dB

        with torch.no_grad():
            for i, (inputs, label, E_inc_raw) in enumerate(val_loader):
                inputs, label, E_inc_raw = inputs.to(device), label.to(device), E_inc_raw.to(device)
                chi = inputs[:, 0:1, :, :] * 3.0
                E_inc = inputs[:, 2:4, :, :]  # 你原来就有，保留（不影响误差统计）

                out = model(inputs, E_inc_raw, chi, label)
                pred = out["PL_pred"]

                bs = inputs.size(0)
                count += bs

                # ===== 训练目标（验证版）：按样本加权，避免最后小 batch 偏差 =====
                val_loss_total += out["loss_total"].item() * bs
                val_loss_data += out["loss_data"].item() * bs
                val_loss_phy += out["loss_phy"].item() * bs

                # ===== 归一化空间误差：per-image =====
                mse = torch.mean((pred - label) ** 2, dim=[1, 2, 3])  # (B,)
                mae = torch.mean(torch.abs(pred - label), dim=[1, 2, 3])  # (B,)
                rmse = torch.sqrt(mse)  # (B,)

                val_mse += mse.sum().item()
                val_rmse += rmse.sum().item()
                val_mae += mae.sum().item()

                # ===== dB 空间误差 =====
                pred_db = pred * scale + PL_trnc
                label_db = label * scale + PL_trnc

                rmse_db = torch.sqrt(torch.mean((pred_db - label_db) ** 2, dim=[1, 2, 3]))  # (B,)
                mae_db = torch.mean(torch.abs(pred_db - label_db), dim=[1, 2, 3])  # (B,)

                val_rmse_db += rmse_db.sum().item()
                val_mae_db += mae_db.sum().item()

                # ===== 每 epoch 前2个 batch 可视化 =====
                if i < 2:
                    visualize_batch(inputs, label, out["PL_init"], out["PL_pred"],
                                    step=(epoch * 1000 + i), phase='val')

        # ===== 均值（全部按样本数平均）=====
        val_mse /= count
        val_rmse /= count
        val_mae /= count
        val_rmse_db /= count
        val_mae_db /= count

        val_loss_total /= count
        val_loss_data /= count
        val_loss_phy /= count

        val_msg = (
            f"Epoch {epoch:02d} | "
            f"ValMSE={val_mse:.6f}, ValRMSE={val_rmse:.6f}, ValMAE={val_mae:.6f}, "
            f"MAE(dB)={val_mae_db:.3f}, RMSE(dB)={val_rmse_db:.3f} | "
            f"ValLossTotal={val_loss_total:.6f} (Data={val_loss_data:.6f}, Phy={val_loss_phy:.6f})"
        )
        print(val_msg)
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(val_msg + "\n")

        # ====== best：用 ValMSE（更可比）=====
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            # 断点里仍然存 best_val_mse（下面 ckpt.save 的 best 参数也改成 best_val_mse）
            ckpt.save(epoch, step, best_val_mse, model, optimizer, scheduler, scaler)
            torch.save(model.state_dict(), model_save_path)

            best_msg = f"--> Saved new best model at epoch {epoch:02d} (ValMSE={val_mse:.6f})"
            print(best_msg)
            with open(train_log_path, "a", encoding="utf-8") as f:
                f.write(best_msg + "\n")

        scheduler.step()
        lr_msg = f"--> LR decayed to {optimizer.param_groups[0]['lr']:.6e}"
        print(lr_msg)
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(lr_msg + "\n")

        # ====== 每轮保存 gc 信息（离线分析用）=====
        gc_pack = {
            "epoch": int(epoch),
            "step": int(step),

            # 主对比指标
            "val_mse": float(val_mse),
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),

            # dB 指标
            "val_mae_db": float(val_mae_db),
            "val_rmse_db": float(val_rmse_db),

            # 训练目标监控
            "val_loss_total": float(val_loss_total),
            "val_loss_data": float(val_loss_data),
            "val_loss_phy": float(val_loss_phy),

            # best 记录（与保存逻辑一致）
            "best_val_mse": float(best_val_mse),

            # 学习率
            "lr": float(optimizer.param_groups[0]["lr"]),

            # 模型快照
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }
        torch.save(gc_pack, gc_snap_dir / f"epoch_{epoch:04d}.pt")

    print(f"\nTraining finished. Best validation MSE = {best_val_mse:.6f}")
    print(f"Best model saved to: {model_save_path}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()