import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"      # 数学符号更像论文
mpl.rcParams["axes.unicode_minus"] = False     # 负号正常显示

from lib.loaders import Dataset_RadioMapSeer
from lib.modules import PhysicsInformedModel
import math


def main():
    # ===================== Config =====================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "./models/best_model_6.pt"
    save_dir = "./inference_results_6"
    vis_dir = os.path.join(save_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    batch_size = 16
    img_size = 256

    NUM_VIS = 10                 # 随机保存 10 张图
    SAVE_FULL_OUTPUT = False      #  是否保存 predictions / labels

    DB_MIN = -147.0
    DB_MAX = -47.84
    scale = DB_MAX - DB_MIN

    # ===================== Dataset =====================
    test_set = Dataset_RadioMapSeer(
        phase="test",
        img_size=img_size,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


    vis_samples = {"87_17", "99_42", "102_6", "121_55", "138_3", "156_29", "166_45", "189_26","379_33","387_53",
                   "418_21","504_28", "510_53","555_47","561_1", "561_9", "561_12", "561_18",
                   "561_34", "561_42", "561_47","561_51", "561_56", "561_67"}


    # ===================== Model =====================
    k = 2 * math.pi * 5.9e9 / 3e8
    dx = 0.01
    model = PhysicsInformedModel(
        k, dx, lambda_d=1.0
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ===================== Metrics =====================
    mse_list = []
    mae_list = []
    mae_db_list = []
    rmse_db_list = []
    r2_list = []
    preds_all = []
    labels_all = []

    # ===================== Inference =====================
    global_idx = 0

    with torch.no_grad():
        for batch_idx, (inputs, label, E_inc_raw,sample_ids ) in enumerate(test_loader):
            inputs = inputs.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            E_inc_raw = E_inc_raw.to(device, non_blocking=True)

            chi = inputs[:, 0:1, :, :] * 3.0
            out = model(inputs, E_inc_raw, chi, label=None)
            pred = out["PL_pred"]

            # ---- normalized metrics ----
            mse = torch.mean((pred - label) ** 2, dim=[1, 2, 3])
            mae = torch.mean(torch.abs(pred - label), dim=[1, 2, 3])
            # ---- R^2  ----
            pred_flat = pred.view(pred.size(0), -1)
            label_flat = label.view(label.size(0), -1)

            ss_res = torch.sum((label_flat - pred_flat) ** 2, dim=1)
            ss_tot = torch.sum((label_flat - label_flat.mean(dim=1, keepdim=True)) ** 2, dim=1)

            r2_norm = 1.0 - ss_res / (ss_tot + 1e-8)

            # ---- dB metrics ----
            pred_db = pred * scale + DB_MIN
            label_db = label * scale + DB_MIN

            rmse_db = torch.sqrt(
                torch.mean((pred_db - label_db) ** 2, dim=[1, 2, 3])
            )
            mae_db = torch.mean(
                torch.abs(pred_db - label_db), dim=[1, 2, 3]
            )

            mse_list.append(mse.cpu())
            mae_list.append(mae.cpu())
            mae_db_list.append(mae_db.cpu())
            rmse_db_list.append(rmse_db.cpu())
            r2_list.append(r2_norm.cpu())

            # ---- optional full dump ----
            if SAVE_FULL_OUTPUT:
                preds_all.append(pred.cpu())
                labels_all.append(label.cpu())
           
            # ---- color_visualization (fixed, not optional) ----
            B = pred.size(0)
            for i in range(B):
                sid = sample_ids[i]
                if sid  in vis_samples:  # if global_idx in vis_indices:
                    p_db = pred_db[i, 0].cpu().numpy()
                    l_db = label_db[i, 0].cpu().numpy()
                    e_db = np.abs(p_db - l_db)

                    #vmin = min(l_db.min(), p_db.min())
                    #vmax = max(l_db.max(), p_db.max())
                    vmin = DB_MIN
                    vmax = DB_MAX

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                    # 坐标设置：0-256，原点在左下角
                    H, W = l_db.shape  # 一般是 256,256
                    extent = [0, W, 0, H]  # 显示为 0~256
                    ticks = [0, 50, 100, 150, 200, 250]

                    im0 = axs[0].imshow(l_db, cmap="viridis", vmin=vmin, vmax=vmax,
                                        origin="lower", extent=extent)
                    axs[0].set_title("Label (dB)")
                    axs[0].set_xticks(ticks)
                    axs[0].set_yticks(ticks)
                    axs[0].set_xlim(0, W)
                    axs[0].set_ylim(0, H)
                    axs[0].set_xlabel("x (m)")
                    axs[0].set_ylabel("y (m)")
                    axs[1].set_xlabel("x (m)")
                    axs[1].set_ylabel("y (m)")
                    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

                    im1 = axs[1].imshow(p_db, cmap="viridis", vmin=vmin, vmax=vmax,
                                        origin="lower", extent=extent)
                    axs[1].set_title("Prediction (dB)")
                    axs[1].set_xticks(ticks)
                    axs[1].set_yticks(ticks)
                    axs[1].set_xlim(0, W)
                    axs[1].set_ylim(0, H)
                    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

                    im2 = axs[2].imshow(e_db, cmap="inferno",
                                        origin="lower", extent=extent)
                    axs[2].set_title("Abs Error (dB)")
                    axs[2].set_xticks(ticks)
                    axs[2].set_yticks(ticks)
                    axs[2].set_xlim(0, W)
                    axs[2].set_ylim(0, H)
                    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f"{sid}.png"), dpi=300)

                    plt.close(fig)

                global_idx += 1

    # ===================== Aggregate Metrics =====================
    mse = torch.cat(mse_list).mean().item()
    mae = torch.cat(mae_list).mean().item()
    mae_db = torch.cat(mae_db_list).mean().item()
    rmse_db = torch.cat(rmse_db_list).mean().item()
    r2 = torch.cat(r2_list).mean().item()

    print("====== Test Results ======")
    print(f"MSE (norm) : {mse:.6f}")
    print(f"MAE (norm) : {mae:.6f}")
    print(f"MAE (dB)   : {mae_db:.3f} dB")
    print(f"RMSE (dB)  : {rmse_db:.3f} dB")
    print(f"R2         : {r2:.4f}")

    # ===================== Save =====================
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "MSE": mse,
            "MAE": mae,
            "MAE_dB": mae_db,
            "RMSE_D B": rmse_db,
            "R2": r2,
        },
        os.path.join(save_dir, "metrics.pt"),
    )

    if SAVE_FULL_OUTPUT:
        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        np.save(os.path.join(save_dir, "predictions.npy"), preds_all.numpy())
        np.save(os.path.join(save_dir, "labels.npy"), labels_all.numpy())

    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
