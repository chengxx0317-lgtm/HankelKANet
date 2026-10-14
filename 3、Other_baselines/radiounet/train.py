import os
import json
import math
import random
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import tensorflow as tf
from tensorflow import keras

from utils  import build_radiounet


SEED = 2025
IMG_SIZE = 256
NUM_TX = 80

# TODO: set this to your RadioMapSeer root directory
DATASET_DIR = r"D:/PycharmProjects/PythonProjects/CXX/PEFNet/data/RadioMapSeer/"

RUN_DIR = "./radiounet_baseline_run"
MODEL_PATH = os.path.join(RUN_DIR, "radiounet_baseline_best.keras")
HISTORY_JSON = os.path.join(RUN_DIR, "history.json")
LOSS_FIG = os.path.join(RUN_DIR, "loss_curve.png")
META_JSON = os.path.join(RUN_DIR, "meta.json")

BATCH_SIZE = 32
EPOCHS = 250
PATIENCE = 50
MIN_DELTA = 1e-6

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-6
GRAD_CLIP = 1.0

BASE_FILTERS = 32


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


def load_one_sample(map_ind: int, tx_idx: int):
    building_path = os.path.join(DATASET_DIR, "png/buildings_complete", f"{map_ind}.png")
    tx_path = os.path.join(DATASET_DIR, "png/antennas", f"{map_ind}_{tx_idx}.png")
    gain_path = os.path.join(DATASET_DIR, "gain/DPM", f"{map_ind}_{tx_idx}.png")

    bld = read_single_channel(building_path)
    tx = read_single_channel(tx_path)
    gain = read_single_channel(gain_path)

    bld = resize_like_dataset(bld, IMG_SIZE, is_label=False)
    tx = resize_like_dataset(tx, IMG_SIZE, is_label=False)
    gain = resize_like_dataset(gain, IMG_SIZE, is_label=True)

    r_map = compute_r_map(tx)

    x = np.stack([bld.astype(np.float32), r_map.astype(np.float32)], axis=-1)
    y = gain.astype(np.float32)[..., None]
    return x, y


class RadioMapSequence(keras.utils.Sequence):
    def __init__(self, sample_refs: List[Tuple[int, int]], batch_size: int, shuffle: bool):
        self.sample_refs = list(sample_refs)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.indices = np.arange(len(self.sample_refs))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.sample_refs) / self.batch_size)

    def __getitem__(self, idx):
        batch_inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        xs = []
        ys = []
        for bi in batch_inds:
            map_ind, tx_idx = self.sample_refs[int(bi)]
            x, y = load_one_sample(map_ind, tx_idx)
            xs.append(x)
            ys.append(y)
        return np.stack(xs, axis=0), np.stack(ys, axis=0)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def cosine_lr_schedule(epoch, lr):
    return ETA_MIN + 0.5 * (LEARNING_RATE - ETA_MIN) * (
        1.0 + np.cos(np.pi * epoch / EPOCHS)
    )
def main():
    seed_everything(SEED)
    enable_gpu_memory_growth()
    os.makedirs(RUN_DIR, exist_ok=True)

    train_maps, val_maps, _ = split_map_indices()
    train_refs = [(int(m), int(t)) for m in train_maps for t in range(NUM_TX)]
    val_refs = [(int(m), int(t)) for m in val_maps for t in range(NUM_TX)]

    train_seq = RadioMapSequence(train_refs, batch_size=BATCH_SIZE, shuffle=True)
    val_seq = RadioMapSequence(val_refs, batch_size=BATCH_SIZE, shuffle=False)

    model = build_radiounet(
        input_shape=(IMG_SIZE, IMG_SIZE, 2),
        base_filters=BASE_FILTERS,
        output_activation="sigmoid",
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            global_clipnorm=GRAD_CLIP,
        ),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.LearningRateScheduler(
            cosine_lr_schedule,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2, ensure_ascii=False)

    meta = {
        "seed": SEED,
        "img_size": IMG_SIZE,
        "num_tx": NUM_TX,
        "dataset_dir": DATASET_DIR,
        "input_channels": ["building", "r_map"],
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "base_filters": BASE_FILTERS,
        "loss": "mse",
        "split": "500/100/100 maps, 80 tx each",
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_FIG, dpi=200)
    plt.close()

    print("Training finished.")
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"History saved to   : {HISTORY_JSON}")
    print(f"Plot saved to      : {LOSS_FIG}")


if __name__ == "__main__":
    main()
