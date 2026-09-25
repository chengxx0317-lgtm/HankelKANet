import os
import json
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from skimage import io, transform
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False

SEED = 2025
IMG_SIZE = 256
NUM_TX = 80

# TODO: set this to your RadioMapSeer root directory
DATASET_DIR = r"D:/PycharmProjects/PythonProjects/CXX/PEFNet/data/RadioMapSeer/"

MODEL_PATH = "./ann_baseline_run/ann_baseline_best.keras"
SAVE_DIR = "./ann_baseline_inference"
VIS_DIR = os.path.join(SAVE_DIR, "vis")
os.makedirs(VIS_DIR, exist_ok=True)

DB_MIN = -147.0
DB_MAX = -47.84
DB_SCALE = DB_MAX - DB_MIN
PRED_BATCH = 8192

vis_samples = {
    "87_17", "99_42", "102_6", "121_55", "138_3", "156_29",
    "166_45", "189_26", "379_33", "387_53", "418_21", "504_28",
    "510_53", "555_47", "561_1", "561_9", "561_12", "561_18",
    "561_34", "561_42", "561_47", "561_51", "561_56", "561_67"
}
SAVE_FULL_OUTPUT = False


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_gpu_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def split_map_indices():
    all_inds = np.arange(0, 700)
    np.random.seed(42)
    np.random.shuffle(all_inds)
    train_maps = all_inds[0:500]
    val_maps = all_inds[500:600]
    test_maps = all_inds[600:700]
    return train_maps, val_maps, test_maps


def read_single_channel(path: str) -> np.ndarray:
    arr = io.imread(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def resize_like_dataset(arr: np.ndarray, img_size: int, is_label: bool) -> np.ndarray:
    if arr.shape[0] != img_size or arr.shape[1] != img_size:
        order = 1 if is_label else 0
        arr = transform.resize(
            arr,
            (img_size, img_size),
            order=order,
            preserve_range=True,
            anti_aliasing=False,
        )
    return arr.astype(np.float32)


def locate_tx(tx_mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(tx_mask > 0)
    if len(xs) == 0:
        y0, x0 = tx_mask.shape[0] // 2, tx_mask.shape[1] // 2
    else:
        y0, x0 = int(ys[0]), int(xs[0])
    return y0, x0


def compute_r_map(tx_mask: np.ndarray) -> np.ndarray:
    h, w = tx_mask.shape
    y0, x0 = locate_tx(tx_mask)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2).astype(np.float32) + 1e-8
    r = (r - r.min()) / (r.max() - r.min() + 1e-9)
    return r.astype(np.float32)


def load_sample_arrays(map_ind: int, tx_idx: int, img_size: int, dataset_dir: str):
    building_path = os.path.join(dataset_dir, "png/buildings_complete", f"{map_ind}.png")
    tx_path = os.path.join(dataset_dir, "png/antennas", f"{map_ind}_{tx_idx}.png")
    gain_path = os.path.join(dataset_dir, "gain/DPM", f"{map_ind}_{tx_idx}.png")

    bld = read_single_channel(building_path)
    tx = read_single_channel(tx_path)
    gain = read_single_channel(gain_path)

    bld = resize_like_dataset(bld, img_size, is_label=False)
    tx = resize_like_dataset(tx, img_size, is_label=False)
    gain = resize_like_dataset(gain, img_size, is_label=True)

    return bld, tx, gain


def build_features_and_label_image(map_ind: int, tx_idx: int):
    bld, tx, gain = load_sample_arrays(map_ind, tx_idx, IMG_SIZE, DATASET_DIR)

    h, w = bld.shape
    tx_row, tx_col = locate_tx(tx)
    r_map = compute_r_map(tx)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    feats = np.stack(
        [
            yy.astype(np.float32) / max(h - 1, 1),
            xx.astype(np.float32) / max(w - 1, 1),
            np.full((h, w), tx_row / max(h - 1, 1), dtype=np.float32),
            np.full((h, w), tx_col / max(w - 1, 1), dtype=np.float32),
            bld.astype(np.float32),
            r_map.astype(np.float32),
        ],
        axis=-1,
    )

    feats = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    return feats, gain.astype(np.float32)


def save_vis(sid: str, label_db: np.ndarray, pred_db: np.ndarray):
    err_db = np.abs(pred_db - label_db)

    vmin = DB_MIN
    vmax = DB_MAX

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    H, W = label_db.shape
    extent = [0, W, 0, H]
    ticks = [0, 50, 100, 150, 200, 250]

    im0 = axs[0].imshow(
        label_db,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=extent,
    )
    axs[0].set_title("Label (dB)")
    axs[0].set_xticks(ticks)
    axs[0].set_yticks(ticks)
    axs[0].set_xlim(0, W)
    axs[0].set_ylim(0, H)
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("y (m)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(
        pred_db,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=extent,
    )
    axs[1].set_title("Prediction (dB)")
    axs[1].set_xticks(ticks)
    axs[1].set_yticks(ticks)
    axs[1].set_xlim(0, W)
    axs[1].set_ylim(0, H)
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("y (m)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(
        err_db,
        cmap="inferno",
        origin="lower",
        extent=extent,
    )
    axs[2].set_title("Abs Error (dB)")
    axs[2].set_xticks(ticks)
    axs[2].set_yticks(ticks)
    axs[2].set_xlim(0, W)
    axs[2].set_ylim(0, H)
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{sid}.png"), dpi=300)
    plt.close(fig)


def main():
    seed_everything(SEED)
    enable_gpu_memory_growth()

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    model = load_model(MODEL_PATH)

    _, _, test_maps = split_map_indices()
    test_refs = [(int(m), int(t)) for m in test_maps for t in range(NUM_TX)]


    mse_list = []
    mae_list = []
    mae_db_list = []
    rmse_db_list = []
    r2_list = []

    preds_all = []
    labels_all = []

    for global_idx, (map_ind, tx_idx) in enumerate(test_refs):
        feats, label_img = build_features_and_label_image(map_ind, tx_idx)
        pred_norm = model.predict(feats, batch_size=PRED_BATCH, verbose=0).reshape(IMG_SIZE, IMG_SIZE)
        pred_norm = np.clip(pred_norm, 0.0, 1.0)

        label_norm = label_img.astype(np.float32)

        pred_db = pred_norm * DB_SCALE + DB_MIN
        label_db = label_norm * DB_SCALE + DB_MIN

        mse = np.mean((pred_norm - label_norm) ** 2, dtype=np.float64)
        mae = np.mean(np.abs(pred_norm - label_norm), dtype=np.float64)
        rmse_db = np.sqrt(np.mean((pred_db - label_db) ** 2, dtype=np.float64))
        mae_db = np.mean(np.abs(pred_db - label_db), dtype=np.float64)

        pred_flat = pred_norm.reshape(-1)
        label_flat = label_norm.reshape(-1)
        ss_res = np.sum((label_flat - pred_flat) ** 2, dtype=np.float64)
        ss_tot = np.sum((label_flat - label_flat.mean()) ** 2, dtype=np.float64)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        mse_list.append(mse)
        mae_list.append(mae)
        mae_db_list.append(mae_db)
        rmse_db_list.append(rmse_db)
        r2_list.append(r2)

        if SAVE_FULL_OUTPUT:
            preds_all.append(pred_norm.astype(np.float32))
            labels_all.append(label_norm.astype(np.float32))

        sid = f"{map_ind}_{tx_idx}"
        if sid in vis_samples:
            save_vis(sid, label_db, pred_db)
    metrics = {
        "MSE": float(np.mean(mse_list)),
        "MAE": float(np.mean(mae_list)),
        "MAE_dB": float(np.mean(mae_db_list)),
        "RMSE_dB": float(np.mean(rmse_db_list)),
        "R2": float(np.mean(r2_list)),
        "seed": int(SEED),
        "img_size": int(IMG_SIZE),
        "num_test_samples": int(len(test_refs)),
    }

    torch.save(metrics, os.path.join(SAVE_DIR, "metrics.pt"))
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if SAVE_FULL_OUTPUT:
        np.save(os.path.join(SAVE_DIR, "predictions.npy"), np.stack(preds_all, axis=0))
        np.save(os.path.join(SAVE_DIR, "labels.npy"), np.stack(labels_all, axis=0))

    print("====== ANN Baseline Test Results ======")
    print(f"MSE (norm) : {metrics['MSE']:.6f}")
    print(f"MAE (norm) : {metrics['MAE']:.6f}")
    print(f"MAE (dB)   : {metrics['MAE_dB']:.3f} dB")
    print(f"RMSE (dB)  : {metrics['RMSE_dB']:.3f} dB")
    print(f"R2         : {metrics['R2']:.4f}")
    print(f"Results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
