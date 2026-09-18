import os
import json
import random
from itertools import product
from typing import List, Tuple

import numpy as np
from skimage import io, transform
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


SEED = 2025
IMG_SIZE = 256
NUM_TX = 80

# TODO: set this to your RadioMapSeer root directory
DATASET_DIR = r"./RadioMapSeer/"

RUN_DIR = "./dtr_baseline_run"
MODEL_PATH = os.path.join(RUN_DIR, "dtr_baseline_best.joblib")
SEARCH_JSON = os.path.join(RUN_DIR, "search_results.json")
META_JSON = os.path.join(RUN_DIR, "meta.json")

# Sampling setup for train/val construction
TRAIN_REF_SUBSET = 40000
VAL_REF_SUBSET = 8000

PIXELS_PER_REF_TRAIN = 32
PIXELS_PER_REF_VAL = 32

# Hyperparameter candidates for validation-based model selection
MAX_DEPTH_CANDIDATES = [20, 30, 40, None]
MIN_SAMPLES_LEAF_CANDIDATES = [1, 10, 50]
MIN_SAMPLES_SPLIT_CANDIDATES = [2, 20]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


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


def build_feature_image_and_label(map_ind: int, tx_idx: int, img_size: int, dataset_dir: str):
    bld, tx, gain = load_sample_arrays(map_ind, tx_idx, img_size, dataset_dir)

    h, w = bld.shape
    tx_row, tx_col = locate_tx(tx)
    r_map = compute_r_map(tx)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    feat_img = np.stack(
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
    return feat_img.astype(np.float32), gain.astype(np.float32)


def sample_tabular_dataset(
    sample_refs: List[Tuple[int, int]],
    num_ref_subset: int,
    pixels_per_ref: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if num_ref_subset < len(sample_refs):
        choose_idx = rng.choice(len(sample_refs), size=num_ref_subset, replace=False)
        chosen_refs = [sample_refs[int(i)] for i in choose_idx]
    else:
        chosen_refs = list(sample_refs)

    x_all = []
    y_all = []

    for map_ind, tx_idx in chosen_refs:
        feat_img, label_img = build_feature_image_and_label(
            map_ind=map_ind,
            tx_idx=tx_idx,
            img_size=IMG_SIZE,
            dataset_dir=DATASET_DIR,
        )

        feat = feat_img.reshape(-1, feat_img.shape[-1])
        label = label_img.reshape(-1)

        n = feat.shape[0]
        if pixels_per_ref < n:
            take = rng.choice(n, size=pixels_per_ref, replace=False)
            feat = feat[take]
            label = label[take]

        x_all.append(feat.astype(np.float32))
        y_all.append(label.astype(np.float32))

    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return x, y


def evaluate_regression(model, x: np.ndarray, y: np.ndarray):
    pred = model.predict(x).astype(np.float32)
    pred = np.clip(pred, 0.0, 1.0)

    mse = float(mean_squared_error(y, pred))
    mae = float(mean_absolute_error(y, pred))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


def main():
    seed_everything(SEED)
    os.makedirs(RUN_DIR, exist_ok=True)

    train_maps, val_maps, _ = split_map_indices()
    train_refs = [(int(m), int(t)) for m in train_maps for t in range(NUM_TX)]
    val_refs = [(int(m), int(t)) for m in val_maps for t in range(NUM_TX)]

    print("Sampling train subset...")
    x_train, y_train = sample_tabular_dataset(
        sample_refs=train_refs,
        num_ref_subset=TRAIN_REF_SUBSET,
        pixels_per_ref=PIXELS_PER_REF_TRAIN,
        seed=SEED,
    )
    print(f"Train sampled shape: X={x_train.shape}, y={y_train.shape}")

    print("Sampling val subset...")
    x_val, y_val = sample_tabular_dataset(
        sample_refs=val_refs,
        num_ref_subset=VAL_REF_SUBSET,
        pixels_per_ref=PIXELS_PER_REF_VAL,
        seed=SEED + 123,
    )
    print(f"Val sampled shape: X={x_val.shape}, y={y_val.shape}")

    search_results = []
    best_model = None
    best_cfg = None
    best_val_mse = float("inf")

    total_trials = len(MAX_DEPTH_CANDIDATES) * len(MIN_SAMPLES_LEAF_CANDIDATES) * len(MIN_SAMPLES_SPLIT_CANDIDATES)
    trial_id = 0

    for max_depth, min_samples_leaf, min_samples_split in product(
        MAX_DEPTH_CANDIDATES,
        MIN_SAMPLES_LEAF_CANDIDATES,
        MIN_SAMPLES_SPLIT_CANDIDATES,
    ):
        trial_id += 1
        print(f"[{trial_id}/{total_trials}] Training DTR with "
              f"max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, min_samples_split={min_samples_split}")

        model = DecisionTreeRegressor(
            criterion="squared_error",
            splitter="best",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=SEED,
        )
        model.fit(x_train, y_train)

        train_mse, train_mae, train_rmse = evaluate_regression(model, x_train, y_train)
        val_mse, val_mae, val_rmse = evaluate_regression(model, x_val, y_val)

        item = {
            "max_depth": max_depth,
            "min_samples_leaf": int(min_samples_leaf),
            "min_samples_split": int(min_samples_split),
            "train_mse": float(train_mse),
            "train_mae": float(train_mae),
            "train_rmse": float(train_rmse),
            "val_mse": float(val_mse),
            "val_mae": float(val_mae),
            "val_rmse": float(val_rmse),
        }
        search_results.append(item)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_cfg = item
            best_model = model
            joblib.dump(best_model, MODEL_PATH)
            print(f"  -> New best model saved. val_mse={val_mse:.6f}")

    with open(SEARCH_JSON, "w", encoding="utf-8") as f:
        json.dump(search_results, f, indent=2, ensure_ascii=False)

    meta = {
        "seed": SEED,
        "img_size": IMG_SIZE,
        "num_tx": NUM_TX,
        "dataset_dir": DATASET_DIR,
        "input_features": [
            "row_norm",
            "col_norm",
            "tx_row_norm",
            "tx_col_norm",
            "building_value",
            "r_map",
        ],
        "train_ref_subset": TRAIN_REF_SUBSET,
        "val_ref_subset": VAL_REF_SUBSET,
        "pixels_per_ref_train": PIXELS_PER_REF_TRAIN,
        "pixels_per_ref_val": PIXELS_PER_REF_VAL,
        "best_config": best_cfg,
        "selection_metric": "val_mse",
        "model_type": "DecisionTreeRegressor",
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Training finished.")
    print(f"Best model saved to : {MODEL_PATH}")
    print(f"Best val config     : {best_cfg}")
    print(f"Search results saved: {SEARCH_JSON}")
    print(f"Meta saved to       : {META_JSON}")


if __name__ == "__main__":
    main()
