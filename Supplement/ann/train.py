import os
import json
import random
import math
from typing import List, Tuple

import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


SEED = 2025
IMG_SIZE = 256
NUM_TX = 80

# TODO: set this to your RadioMapSeer root directory
DATASET_DIR = r"./RadioMapSeer/"

RUN_DIR = "./ann_baseline_run"
MODEL_PATH = os.path.join(RUN_DIR, "ann_baseline_best.keras")
HISTORY_JSON = os.path.join(RUN_DIR, "history.json")
LOSS_FIG = os.path.join(RUN_DIR, "ann_loss.png")
META_JSON = os.path.join(RUN_DIR, "meta.json")

EPOCHS = 250
PATIENCE = 50
MIN_DELTA = 1e-6

TRAIN_STEPS_PER_EPOCH = 400
SAMPLES_PER_BATCH = 32
PIXELS_PER_SAMPLE_TRAIN = 4096

PIXELS_PER_SAMPLE_VAL = 160

LEARNING_RATE = 4e-3
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-6
GRAD_CLIP = 1.0


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


def build_features_and_labels(
    map_ind: int,
    tx_idx: int,
    img_size: int,
    dataset_dir: str,
):
    bld, tx, gain = load_sample_arrays(map_ind, tx_idx, img_size, dataset_dir)

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
    labels = gain.reshape(-1, 1).astype(np.float32)
    return feats, labels


class RandomPixelSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        sample_refs: List[Tuple[int, int]],
        dataset_dir: str,
        img_size: int,
        steps_per_epoch: int,
        samples_per_batch: int,
        pixels_per_sample: int,
        seed: int,
    ):
        self.sample_refs = list(sample_refs)
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.steps_per_epoch = int(steps_per_epoch)
        self.samples_per_batch = int(samples_per_batch)
        self.pixels_per_sample = int(pixels_per_sample)
        self.seed = int(seed)

        self.epoch = 0
        self.refs_per_epoch = self.steps_per_epoch * self.samples_per_batch

        if self.refs_per_epoch > len(self.sample_refs):
            raise ValueError(
                "refs_per_epoch cannot exceed the number of training refs "
                "when sampling without replacement."
            )

        self._refresh_epoch_ids()

    def _refresh_epoch_ids(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch_ids = rng.choice(
            len(self.sample_refs),
            size=self.refs_per_epoch,
            replace=False,
        )

    def __len__(self):
        return self.steps_per_epoch

    def on_epoch_end(self):
        self.epoch += 1
        self._refresh_epoch_ids()

    def __getitem__(self, idx):
        start = idx * self.samples_per_batch
        end = start + self.samples_per_batch
        chosen_ids = self.epoch_ids[start:end]

        rng = np.random.default_rng(
            self.seed + self.epoch * 100000 + idx
        )

        batch_x = []
        batch_y = []

        for sid in chosen_ids:
            map_ind, tx_idx = self.sample_refs[int(sid)]

            feats, labels = build_features_and_labels(
                map_ind=map_ind,
                tx_idx=tx_idx,
                img_size=self.img_size,
                dataset_dir=self.dataset_dir,
            )

            n = feats.shape[0]

            if self.pixels_per_sample < n:
                take = rng.choice(
                    n,
                    size=self.pixels_per_sample,
                    replace=False,
                )
                feats = feats[take]
                labels = labels[take]

            batch_x.append(feats)
            batch_y.append(labels)

        x = np.concatenate(batch_x, axis=0).astype(np.float32)
        y = np.concatenate(batch_y, axis=0).astype(np.float32)

        return x, y

def build_fixed_validation_dataset(
    sample_refs: List[Tuple[int, int]],
    dataset_dir: str,
    img_size: int,
    pixels_per_sample: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    x_all = []
    y_all = []

    for map_ind, tx_idx in sample_refs:
        feats, labels = build_features_and_labels(
            map_ind=map_ind,
            tx_idx=tx_idx,
            img_size=img_size,
            dataset_dir=dataset_dir,
        )

        n = feats.shape[0]

        if pixels_per_sample < n:
            take = rng.choice(
                n,
                size=pixels_per_sample,
                replace=False,
            )
            feats = feats[take]
            labels = labels[take]

        x_all.append(feats.astype(np.float32))
        y_all.append(labels.astype(np.float32))

    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    return x, y
def cosine_lr_schedule(epoch, lr):
    return ETA_MIN + 0.5 * (LEARNING_RATE - ETA_MIN) * (
        1.0 + np.cos(np.pi * epoch / EPOCHS)
    )
def build_ann(input_dim: int = 6):
    inp = layers.Input(shape=(input_dim,), name="pixel_features")
    x = layers.Dense(300, activation="relu")(inp)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(10, activation="relu")(x)
    out = layers.Dense(1, activation=None)(x)
    model = models.Model(inp, out, name="ANNBaseline")

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            global_clipnorm=GRAD_CLIP,
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def main():
    seed_everything(SEED)
    enable_gpu_memory_growth()

    os.makedirs(RUN_DIR, exist_ok=True)

    train_maps, val_maps, _ = split_map_indices()
    train_refs = [(int(m), int(t)) for m in train_maps for t in range(NUM_TX)]
    val_refs = [(int(m), int(t)) for m in val_maps for t in range(NUM_TX)]

    train_seq = RandomPixelSequence(
        sample_refs=train_refs,
        dataset_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
        samples_per_batch=SAMPLES_PER_BATCH,
        pixels_per_sample=PIXELS_PER_SAMPLE_TRAIN,
        seed=SEED,
    )

    print("Building fixed validation dataset...")

    val_x, val_y = build_fixed_validation_dataset(
        sample_refs=val_refs,
        dataset_dir=DATASET_DIR,
        img_size=IMG_SIZE,
        pixels_per_sample=PIXELS_PER_SAMPLE_VAL,
        seed=SEED + 123,
    )

    print(
        f"Validation dataset: X={val_x.shape}, "
        f"y={val_y.shape}"
    )

    model = build_ann(input_dim=6)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),

        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        LearningRateScheduler(
            cosine_lr_schedule,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_seq,
        validation_data=(val_x, val_y),
        validation_batch_size=16384,
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
        "input_features": [
            "row_norm",
            "col_norm",
            "tx_row_norm",
            "tx_col_norm",
            "building_value",
            "r_map",
        ],
        "train_steps_per_epoch": TRAIN_STEPS_PER_EPOCH,
        "samples_per_batch": SAMPLES_PER_BATCH,
        "pixels_per_sample_train": PIXELS_PER_SAMPLE_TRAIN,
        "pixels_per_sample_val": PIXELS_PER_SAMPLE_VAL,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "loss": "mse",
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
