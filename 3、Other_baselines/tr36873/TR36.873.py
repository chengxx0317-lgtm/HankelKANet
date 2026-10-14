import os
import json
import random
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io, transform
from tqdm import tqdm

from loaders import Dataset_RadioMapSeer
import matplotlib as mpl
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False


DB_MIN = -147.0
DB_MAX = -47.84
DB_SCALE = DB_MAX - DB_MIN

IMG_SIZE = 256
EVAL_PHASE = "test"
SAVE_DIR = "./baseline_tr36873_1_results"
VIS_DIR = os.path.join(SAVE_DIR, "vis")
os.makedirs(VIS_DIR, exist_ok=True)

# TODO: set this to your local RadioMapSeer root directory
DATASET_DIR = r"D:/PycharmProjects/PythonProjects/CXX/PEFNet/data/RadioMapSeer/"
vis_samples = {
    "87_17", "99_42", "102_6", "121_55", "138_3", "156_29",
    "166_45", "189_26", "379_33", "387_53", "418_21", "504_28",
    "510_53", "555_47", "561_1", "561_9", "561_12", "561_18",
    "561_34", "561_42", "561_47", "561_51", "561_56", "561_67"
}

# Radio parameters
FC = 5.9e9
USR_H = 1.5
BS_H = 10.0

# Reproducibility
SEED = 2025
STOCHASTIC = False

# Map-aware options
BUILDING_THRESHOLD = 0.5       # building pixels are assumed > threshold
APPLY_O2I = True
O2I_LOSS_DB = 20.0             # configurable fixed penetration loss term
CLIP_PRED_TO_VALID_DB_RANGE = True


MASK_BUILDINGS_IN_VIS = True
BUILDING_VIS_DB = DB_MIN - 5.0 # only for plotting, NOT for metric computation
SAVE_FULL_OUTPUT = False



def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def read_single_channel(path: str) -> np.ndarray:
    x = io.imread(path).astype(np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    if x.max() > 1.0:
        x /= 255.0
    return x


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


def building_to_masks(building_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indoor_mask = building_map > BUILDING_THRESHOLD
    outdoor_mask = ~indoor_mask
    return indoor_mask, outdoor_mask


def calculate_umi_path_loss_map(
    h: int,
    w: int,
    tx_row: int,
    tx_col: int,
    fc: float = FC,
    usr_h: float = USR_H,
    bs_h: float = BS_H,
    stochastic: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Returns POSITIVE path loss in dB, shape (H, W).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    c = 3e8

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d2d = np.sqrt((yy - tx_row) ** 2 + (xx - tx_col) ** 2).astype(np.float32)
    d2d_safe = np.maximum(d2d, 1e-6)
    d3d = np.sqrt(d2d_safe ** 2 + (bs_h - usr_h) ** 2).astype(np.float32)

    # LOS probability for UMi
    p_los = np.where(
        d2d <= 18.0,
        1.0,
        18.0 / d2d_safe + np.exp(-d2d_safe / 36.0) * (1.0 - 18.0 / d2d_safe),
    )
    p_los = np.clip(p_los, 0.0, 1.0)

    d_bp = 4.0 * (bs_h - 1.0) * (usr_h - 1.0) * fc / c
    fc_ghz = fc * 1e-9

    pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
    pl2 = (
        28.0
        + 40.0 * np.log10(d3d)
        + 20.0 * np.log10(fc_ghz)
        - 9.0 * np.log10(d_bp ** 2 + (bs_h - usr_h) ** 2)
    )
    pl_los = np.where(d2d <= d_bp, pl1, pl2)

    pl_nlos = 22.7 + 36.7 * np.log10(d3d) + 26.0 * np.log10(fc_ghz) - 0.3 * (usr_h - 1.5)
    pl_nlos = np.maximum(pl_los, pl_nlos)

    if stochastic:
        has_los = rng.binomial(1, p_los, size=p_los.shape)
        noise_los = rng.normal(loc=0.0, scale=3.0, size=p_los.shape)
        noise_nlos = rng.normal(loc=0.0, scale=4.0, size=p_los.shape)
        pl = np.where(
            has_los == 1,
            pl_los + noise_los,
            pl_nlos + noise_nlos,
        )
    else:
        pl = p_los * pl_los + (1.0 - p_los) * pl_nlos

    return pl.astype(np.float32)


def apply_o2i_penetration_loss(
    pl_pos_db: np.ndarray,
    indoor_mask: np.ndarray,
    apply_o2i: bool = APPLY_O2I,
    o2i_loss_db: float = O2I_LOSS_DB,
) -> np.ndarray:
    if not apply_o2i:
        return pl_pos_db
    pl = pl_pos_db.copy()
    pl[indoor_mask] += float(o2i_loss_db)
    return pl


def compute_metrics(
    pred_norm: np.ndarray,
    label_norm: np.ndarray,
    pred_db: np.ndarray,
    label_db: np.ndarray,
    mask: np.ndarray | None,
) -> Dict[str, float]:
    if mask is None:
        pn = pred_norm.reshape(-1)
        ln = label_norm.reshape(-1)
        pdb = pred_db.reshape(-1)
        ldb = label_db.reshape(-1)
    else:
        pn = pred_norm[mask].reshape(-1)
        ln = label_norm[mask].reshape(-1)
        pdb = pred_db[mask].reshape(-1)
        ldb = label_db[mask].reshape(-1)

    mse = float(np.mean((pn - ln) ** 2, dtype=np.float64))
    mae = float(np.mean(np.abs(pn - ln), dtype=np.float64))
    rmse_db = float(np.sqrt(np.mean((pdb - ldb) ** 2, dtype=np.float64)))
    mae_db = float(np.mean(np.abs(pdb - ldb), dtype=np.float64))

    ss_res = float(np.sum((ln - pn) ** 2, dtype=np.float64))
    ss_tot = float(np.sum((ln - ln.mean()) ** 2, dtype=np.float64))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

    return {
        "MSE": mse,
        "MAE": mae,
        "MAE_dB": mae_db,
        "RMSE_dB": rmse_db,
        "R2": r2,
        "num_pixels": int(len(ln)),
    }


def append_metric(bucket: dict, prefix: str, item: Dict[str, float]):
    for k, v in item.items():
        key = f"{prefix}_{k}"
        bucket.setdefault(key, []).append(v)


def finalize_metric_lists(bucket: dict) -> dict:
    out = {}
    for k, vals in bucket.items():
        if k.endswith("num_pixels"):
            out[k] = int(np.sum(vals))
        else:
            out[k] = float(np.mean(vals))
    return out


def save_vis(
    sid: str,
    label_db: np.ndarray,
    pred_db: np.ndarray,
    indoor_mask: np.ndarray | None = None,
):
    label_vis = label_db.copy()
    pred_vis = pred_db.copy()
    err_db = np.abs(pred_db - label_db)

    # 保留统一色域
    vmin = DB_MIN
    vmax = DB_MAX

    # 保留“建筑物强制同色”的老逻辑
    if indoor_mask is not None:
        label_vis[indoor_mask] = BUILDING_VIS_DB
        pred_vis[indoor_mask] = BUILDING_VIS_DB
        err_db = err_db.copy()
        err_db[indoor_mask] = 0.0

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    H, W = label_db.shape
    extent = [0, W, 0, H]
    ticks = [0, 50, 100, 150, 200, 250]

    im0 = axs[0].imshow(
        label_vis,
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
        pred_vis,
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
    rng = np.random.default_rng(SEED)

    test_set = Dataset_RadioMapSeer(
        phase=EVAL_PHASE,
        dir_dataset=DATASET_DIR,
        img_size=IMG_SIZE,
    )


    metric_lists = {}

    preds_all = []
    labels_all = []

    for global_idx in tqdm(range(len(test_set)), desc="TR36.873 map-aware baseline"):
        idxr = global_idx // test_set.numTx
        idxc = global_idx % test_set.numTx
        map_ind = int(test_set.map_inds[idxr])

        building_path = os.path.join(test_set.dir_buildings, f"{map_ind}.png")
        tx_path = os.path.join(test_set.dir_Tx, f"{map_ind}_{idxc}.png")
        gain_path = os.path.join(test_set.dir_gain, f"{map_ind}_{idxc}.png")

        bld = read_single_channel(building_path)
        tx = read_single_channel(tx_path)
        gain = read_single_channel(gain_path)

        bld = resize_like_dataset(bld, IMG_SIZE, is_label=False)
        tx = resize_like_dataset(tx, IMG_SIZE, is_label=False)
        gain = resize_like_dataset(gain, IMG_SIZE, is_label=True)

        indoor_mask, outdoor_mask = building_to_masks(bld)
        tx_row, tx_col = locate_tx(tx)

        pl_pos_db = calculate_umi_path_loss_map(
            h=IMG_SIZE,
            w=IMG_SIZE,
            tx_row=tx_row,
            tx_col=tx_col,
            stochastic=STOCHASTIC,
            rng=rng,
        )
        pl_pos_db = apply_o2i_penetration_loss(
            pl_pos_db=pl_pos_db,
            indoor_mask=indoor_mask,
            apply_o2i=APPLY_O2I,
            o2i_loss_db=O2I_LOSS_DB,
        )

        pred_db = -pl_pos_db
        if CLIP_PRED_TO_VALID_DB_RANGE:
            pred_db = np.clip(pred_db, DB_MIN, DB_MAX)

        label_norm = gain.astype(np.float32)
        label_db = label_norm * DB_SCALE + DB_MIN
        pred_norm = (pred_db - DB_MIN) / DB_SCALE
        pred_norm = np.clip(pred_norm, 0.0, 1.0)

        full_metrics = compute_metrics(
            pred_norm=pred_norm,
            label_norm=label_norm,
            pred_db=pred_db,
            label_db=label_db,
            mask=None,
        )
        outdoor_metrics = compute_metrics(
            pred_norm=pred_norm,
            label_norm=label_norm,
            pred_db=pred_db,
            label_db=label_db,
            mask=outdoor_mask,
        )

        append_metric(metric_lists, "full", full_metrics)
        append_metric(metric_lists, "outdoor", outdoor_metrics)

        if SAVE_FULL_OUTPUT:
            preds_all.append(pred_norm.astype(np.float32))
            labels_all.append(label_norm.astype(np.float32))

        sid = f"{map_ind}_{idxc}"
        if sid in vis_samples:
            save_vis(
                sid=sid,
                label_db=label_db,
                pred_db=pred_db,
                indoor_mask=indoor_mask,
            )

    metrics = finalize_metric_lists(metric_lists)
    metrics.update(
        {
            "stochastic": bool(STOCHASTIC),
            "seed": int(SEED),
            "img_size": int(IMG_SIZE),
            "num_test_samples": int(len(test_set)),
            "apply_o2i": bool(APPLY_O2I),
            "o2i_loss_db": float(O2I_LOSS_DB),
            "building_threshold": float(BUILDING_THRESHOLD),

        }
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(metrics, os.path.join(SAVE_DIR, "metrics.pt"))
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if SAVE_FULL_OUTPUT:
        np.save(os.path.join(SAVE_DIR, "predictions.npy"), np.stack(preds_all, axis=0))
        np.save(os.path.join(SAVE_DIR, "labels.npy"), np.stack(labels_all, axis=0))

    print("====== TR 36.873 Map-Aware Baseline Test Results ======")
    print(f"FULL    MSE (norm): {metrics['full_MSE']:.6f}")
    print(f"FULL    MAE (norm): {metrics['full_MAE']:.6f}")
    print(f"FULL    MAE (dB)  : {metrics['full_MAE_dB']:.3f} dB")
    print(f"FULL    RMSE (dB) : {metrics['full_RMSE_dB']:.3f} dB")
    print(f"FULL    R2        : {metrics['full_R2']:.4f}")
    print("------------------------------------------------------")
    print(f"OUTDOOR MSE (norm): {metrics['outdoor_MSE']:.6f}")
    print(f"OUTDOOR MAE (norm): {metrics['outdoor_MAE']:.6f}")
    print(f"OUTDOOR MAE (dB)  : {metrics['outdoor_MAE_dB']:.3f} dB")
    print(f"OUTDOOR RMSE (dB) : {metrics['outdoor_RMSE_dB']:.3f} dB")
    print(f"OUTDOOR R2        : {metrics['outdoor_R2']:.4f}")
    print(f"Results saved to  : {SAVE_DIR}")


if __name__ == "__main__":
    main()
