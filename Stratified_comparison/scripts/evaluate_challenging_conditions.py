import csv
import gc
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stratified_common import (
    DB_MIN,
    DB_SCALE,
    IMG_SIZE,
    StratifiedRadioMapSeerDataset,
    building_density,
    load_scene_los_cache,
    load_thresholds,
    unpack_bool_mask,
)


# ============================================================================
# PyCharm configuration: edit only this block
# ============================================================================
DATASET_DIR = r"D:\PycharmProjects\PythonProjects\CXX\PEFNet\data\RadioMapSeer"
CACHE_DIR = r"./challenging_condition_cache"
RESULT_DIR = r"./challenging_condition_results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_BATCH_SIZE = 4
NUM_WORKERS = 0
DX_METERS = 1.0

MIN_PIXELS_PER_MAP = 10

# Code directories. Change to your actual folders.
BASELINE_CODE_DIR = r"D:\PycharmProjects\PythonProjects\CXX\Hankel_kan_convolutional"
ORDER_CODE_DIR = r"D:\PycharmProjects\PythonProjects\CXX\阶数消融实验"
ANGULAR_CODE_DIR = r"D:\PycharmProjects\PythonProjects\CXX\角向实验"

# ---------------------------------------------------------------------------
MODEL_SPECS = [
    {
        "enabled": True,
        "name": "Radial_{0,1}",
        "kind": "baseline",
        "code_dir": BASELINE_CODE_DIR,
        "checkpoint": r"D:\PycharmProjects\PythonProjects\CXX\Hankel_kan_convolutional\models\hkan_17.pt",
    },

    {
        "enabled": False,
        "name": "Radial_{0,1,2}",
        "kind": "order",
        "code_dir": ORDER_CODE_DIR,
        "orders": (0, 1, 2),
        "checkpoint": r"D:\CHANGE_ME\models_order_ablation\orders_0-1-2_seed0.pt",
    },

    {
        "enabled": False,
        "name": "Angular_|n|<=1",
        "kind": "angular",
        "code_dir": ANGULAR_CODE_DIR,
        "max_order": 1,
        "checkpoint": r"D:\CHANGE_ME\models_angular_ablation\angular_leq1_seed0.pt",
    },
]

# Diagnostics only. They are not Table-3 rows.
SAVE_ALL_PIXELS_DIAGNOSTIC = True
SAVE_OUTDOOR_ALL_DIAGNOSTIC = True


# ============================================================================
# Model loading
# ============================================================================
def ensure_code_dir(code_dir):
    code_dir = str(Path(code_dir).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


def build_model(spec):
    ensure_code_dir(spec["code_dir"])

    if spec["kind"] == "baseline":
        mod = importlib.import_module("modules")
        return mod.HKANNet(base_ch=32, phys_ch=32)

    if spec["kind"] == "order":
        mod = importlib.import_module(
            "modules_order_ablation"
        )
        return mod.HKANNet(
            base_ch=32,
            phys_ch=32,
            orders=tuple(spec["orders"]),
        )

    if spec["kind"] == "angular":
        mod = importlib.import_module("modules_angular")
        return mod.HKANAngularNet(
            base_ch=32,
            phys_ch=32,
            max_order=int(spec["max_order"]),
        )

    raise ValueError(
        f"Unknown model kind: {spec['kind']}"
    )


def extract_state_dict(obj):
    if not isinstance(obj, dict):
        raise ValueError(
            f"Unsupported checkpoint type: {type(obj)}"
        )

    if len(obj) > 0 and all(
        torch.is_tensor(v) for v in obj.values()
    ):
        state = obj
    elif "model" in obj and isinstance(
        obj["model"], dict
    ):
        state = obj["model"]
    elif "state_dict" in obj and isinstance(
        obj["state_dict"], dict
    ):
        state = obj["state_dict"]
    elif "model_state_dict" in obj and isinstance(
        obj["model_state_dict"], dict
    ):
        state = obj["model_state_dict"]
    else:
        raise ValueError(
            "Checkpoint does not contain a recognized "
            "model state_dict."
        )

    if any(
        k.startswith("module.") for k in state.keys()
    ):
        state = {
            (
                k[7:]
                if k.startswith("module.")
                else k
            ): v
            for k, v in state.items()
        }

    return state


def load_model(spec):
    ckpt_path = Path(spec["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found for {spec['name']}:\n"
            f"{ckpt_path}\n"
            f"Edit MODEL_SPECS at the top of this script."
        )

    model = build_model(spec).to(DEVICE)
    raw = torch.load(
        ckpt_path, map_location=DEVICE
    )
    state = extract_state_dict(raw)

    try:
        model.load_state_dict(
            state, strict=True
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"\nStrict loading failed for {spec['name']}.\n"
            f"kind={spec['kind']}\n"
            f"checkpoint={ckpt_path}\n"
            f"Check code version and orders/max_order.\n\n"
            f"{exc}"
        ) from exc

    model.eval()
    return model


# ============================================================================
# Metrics
# ============================================================================
REQUIRED_CONDITIONS = [
    "LOS",
    "NLOS",
    "SHORT_DISTANCE",
    "LONG_DISTANCE",
    "HIGH_BUILDING_DENSITY",
]


def metric_on_mask(
    pred_db,
    label_db,
    mask,
    min_pixels=10,
):
    mask = np.asarray(
        mask, dtype=np.bool_
    )
    n_pixels = int(mask.sum())

    if n_pixels < int(min_pixels):
        return None

    err = pred_db[mask] - label_db[mask]
    mae = float(
        np.mean(np.abs(err))
    )
    rmse = float(
        np.sqrt(np.mean(err * err))
    )

    return {
        "n_pixels": n_pixels,
        "mae_db": mae,
        "rmse_db": rmse,
    }


def append_metric(
    rows,
    model_name,
    sample_id,
    scene_id,
    tx_id,
    condition,
    metric,
):
    if metric is None:
        return

    rows.append(
        {
            "model": model_name,
            "sample_id": sample_id,
            "scene_id": int(scene_id),
            "tx_id": int(tx_id),
            "condition": condition,
            "n_pixels": int(
                metric["n_pixels"]
            ),
            "mae_db": float(
                metric["mae_db"]
            ),
            "rmse_db": float(
                metric["rmse_db"]
            ),
        }
    )


def aggregate_rows(per_map_rows):
    buckets = defaultdict(list)
    model_order = []

    for row in per_map_rows:
        key = (
            row["model"],
            row["condition"],
        )
        buckets[key].append(row)

        if row["model"] not in model_order:
            model_order.append(row["model"])

    condition_order = []

    if SAVE_ALL_PIXELS_DIAGNOSTIC:
        condition_order.append(
            "ALL_PIXELS_DIAGNOSTIC"
        )

    if SAVE_OUTDOOR_ALL_DIAGNOSTIC:
        condition_order.append(
            "OUTDOOR_ALL_DIAGNOSTIC"
        )

    condition_order += REQUIRED_CONDITIONS

    out = []

    for model_name in model_order:
        for condition in condition_order:
            rows = buckets.get(
                (model_name, condition), []
            )

            if not rows:
                continue

            maes = np.asarray(
                [
                    r["mae_db"]
                    for r in rows
                ],
                dtype=np.float64,
            )
            rmses = np.asarray(
                [
                    r["rmse_db"]
                    for r in rows
                ],
                dtype=np.float64,
            )

            out.append(
                {
                    "model": model_name,
                    "condition": condition,
                    "maps_evaluated": len(rows),
                    "total_pixels": int(
                        sum(
                            r["n_pixels"]
                            for r in rows
                        )
                    ),
                    "mae_db_mean": float(
                        maes.mean()
                    ),
                    "mae_db_std_across_maps": float(
                        maes.std(ddof=0)
                    ),
                    "rmse_db_mean": float(
                        rmses.mean()
                    ),
                    "rmse_db_std_across_maps": float(
                        rmses.std(ddof=0)
                    ),
                }
            )

    return out


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class SceneLosCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.scene_id = None
        self.data = None

    def get(self, scene_id, tx_id):
        scene_id = int(scene_id)
        tx_id = int(tx_id)

        if self.scene_id != scene_id:
            self.data = load_scene_los_cache(
                self.cache_dir,
                scene_id,
            )
            self.scene_id = scene_id

        packed = self.data[
            "los_packed"
        ][tx_id]
        shape = self.data["shape"]

        return unpack_bool_mask(
            packed, shape
        )


# ============================================================================
# Evaluate one fixed model
# ============================================================================
def evaluate_one_model(
    spec,
    thresholds,
    loader,
):
    model_name = spec["name"]
    model = load_model(spec)
    los_cache = SceneLosCache(CACHE_DIR)

    short_max = float(
        thresholds[
            "short_distance_max_m"
        ]
    )
    long_min = float(
        thresholds[
            "long_distance_min_m"
        ]
    )
    high_density_min = float(
        thresholds[
            "high_building_density_min"
        ]
    )

    per_map_rows = []
    processed = 0

    print("\n" + "=" * 95)
    print(
        f"Evaluating: {model_name}"
    )
    print(
        f"Kind      : {spec['kind']}"
    )
    print(
        f"Checkpoint: {spec['checkpoint']}"
    )
    print(
        f"Short     : d <= {short_max:.4f} m"
    )
    print(
        f"Long      : d >= {long_min:.4f} m"
    )
    print(
        f"High dens.: density >= "
        f"{high_density_min:.6f}"
    )
    print("=" * 95)

    with torch.inference_mode():
        for batch in loader:
            bld = batch["building"]
            bld_mask = batch[
                "building_mask"
            ]
            r_map = batch["r_map"]
            phi_map = batch["phi_map"]
            label = batch["label"]
            distance_m = batch[
                "distance_m"
            ]

            scene_ids = batch[
                "scene_id"
            ]
            tx_ids = batch["tx_id"]
            sample_ids = batch[
                "sample_id"
            ]
            tx_coords = batch[
                "tx_coord"
            ]

            if spec["kind"] in (
                "baseline",
                "order",
            ):
                inputs = torch.cat(
                    [bld, r_map],
                    dim=1,
                )

            elif spec["kind"] == "angular":
                inputs = torch.cat(
                    [
                        bld,
                        r_map,
                        phi_map,
                    ],
                    dim=1,
                )

            else:
                raise ValueError(
                    spec["kind"]
                )

            inputs = inputs.to(
                DEVICE,
                non_blocking=True,
            )
            label_gpu = label.to(
                DEVICE,
                non_blocking=True,
            )

            # FP32 accuracy evaluation.
            pred = model(inputs)

            pred_db_batch = (
                pred.detach()
                .cpu()
                .numpy()[:, 0]
                * DB_SCALE
                + DB_MIN
            )
            label_db_batch = (
                label_gpu.detach()
                .cpu()
                .numpy()[:, 0]
                * DB_SCALE
                + DB_MIN
            )

            bld_mask_np = (
                bld_mask.numpy()[:, 0]
                .astype(bool)
            )
            distance_np = (
                distance_m.numpy()[:, 0]
            )

            B = pred_db_batch.shape[0]

            for j in range(B):
                scene_id = int(
                    scene_ids[j]
                )
                tx_id = int(
                    tx_ids[j]
                )
                sample_id = str(
                    sample_ids[j]
                )

                pred_db = (
                    pred_db_batch[j]
                )
                label_db = (
                    label_db_batch[j]
                )
                bm = bld_mask_np[j]
                dist = distance_np[j]

                y0 = int(
                    tx_coords[j, 0]
                )
                x0 = int(
                    tx_coords[j, 1]
                )

                # Stratified Rx subsets:
                # outdoor only, excluding Tx.
                outdoor = ~bm
                outdoor[y0, x0] = False

                los = (
                    los_cache.get(
                        scene_id,
                        tx_id,
                    )
                    & outdoor
                )
                nlos = (
                    outdoor
                    & (~los)
                )

                short_mask = (
                    outdoor
                    & (
                        dist
                        <= short_max
                    )
                    & (dist > 0.0)
                )

                long_mask = (
                    outdoor
                    & (
                        dist
                        >= long_min
                    )
                )

                dens = building_density(
                    bm
                )

                if (
                    dens
                    >= high_density_min
                ):
                    high_density_mask = (
                        outdoor
                    )
                else:
                    high_density_mask = (
                        np.zeros_like(
                            outdoor,
                            dtype=bool,
                        )
                    )

                # Whole-map diagnostic:
                # should approximately recover
                # the normal test metric.
                if (
                    SAVE_ALL_PIXELS_DIAGNOSTIC
                ):
                    metric = metric_on_mask(
                        pred_db,
                        label_db,
                        np.ones_like(
                            outdoor,
                            dtype=bool,
                        ),
                        MIN_PIXELS_PER_MAP,
                    )
                    append_metric(
                        per_map_rows,
                        model_name,
                        sample_id,
                        scene_id,
                        tx_id,
                        "ALL_PIXELS_DIAGNOSTIC",
                        metric,
                    )

                if (
                    SAVE_OUTDOOR_ALL_DIAGNOSTIC
                ):
                    metric = metric_on_mask(
                        pred_db,
                        label_db,
                        outdoor,
                        MIN_PIXELS_PER_MAP,
                    )
                    append_metric(
                        per_map_rows,
                        model_name,
                        sample_id,
                        scene_id,
                        tx_id,
                        "OUTDOOR_ALL_DIAGNOSTIC",
                        metric,
                    )

                masks = {
                    "LOS": los,
                    "NLOS": nlos,
                    "SHORT_DISTANCE": short_mask,
                    "LONG_DISTANCE": long_mask,
                    "HIGH_BUILDING_DENSITY": (
                        high_density_mask
                    ),
                }

                for (
                    condition,
                    mask,
                ) in masks.items():
                    metric = (
                        metric_on_mask(
                            pred_db,
                            label_db,
                            mask,
                            MIN_PIXELS_PER_MAP,
                        )
                    )
                    append_metric(
                        per_map_rows,
                        model_name,
                        sample_id,
                        scene_id,
                        tx_id,
                        condition,
                        metric,
                    )

                processed += 1

            if processed % 200 == 0:
                print(
                    f"[{model_name}] "
                    f"{processed}/"
                    f"{len(loader.dataset)} maps"
                )

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return per_map_rows


def print_required_table(
    aggregate,
):
    print("\n" + "=" * 100)
    print(
        "TABLE-3 STYLE SUMMARY "
        "(per-map mean metrics)"
    )
    print("=" * 100)
    print(
        f"{'Model':28s} "
        f"{'Condition':26s} "
        f"{'MAE(dB)':>12s} "
        f"{'RMSE(dB)':>12s} "
        f"{'Maps':>8s} "
        f"{'Pixels':>12s}"
    )
    print("-" * 100)

    for row in aggregate:
        if (
            row["condition"]
            not in REQUIRED_CONDITIONS
        ):
            continue

        print(
            f"{row['model'][:28]:28s} "
            f"{row['condition'][:26]:26s} "
            f"{row['mae_db_mean']:12.6f} "
            f"{row['rmse_db_mean']:12.6f} "
            f"{row['maps_evaluated']:8d} "
            f"{row['total_pixels']:12d}"
        )


def main():
    result_dir = Path(
        RESULT_DIR
    )
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds = load_thresholds(
        CACHE_DIR
    )

    dataset = (
        StratifiedRadioMapSeerDataset(
            dataset_dir=DATASET_DIR,
            phase="test",
            img_size=IMG_SIZE,
            dx=DX_METERS,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    enabled_specs = [
        s
        for s in MODEL_SPECS
        if s.get(
            "enabled", False
        )
    ]

    if not enabled_specs:
        raise RuntimeError(
            "No MODEL_SPECS enabled."
        )

    all_per_map_rows = []

    for spec in enabled_specs:
        rows = evaluate_one_model(
            spec,
            thresholds,
            loader,
        )
        all_per_map_rows.extend(
            rows
        )

    aggregate = aggregate_rows(
        all_per_map_rows
    )

    per_map_path = (
        result_dir
        / "stratified_per_map.csv"
    )
    summary_path = (
        result_dir
        / "stratified_summary.csv"
    )
    metadata_path = (
        result_dir
        / "experiment_metadata.json"
    )

    write_csv(
        per_map_path,
        all_per_map_rows,
        [
            "model",
            "sample_id",
            "scene_id",
            "tx_id",
            "condition",
            "n_pixels",
            "mae_db",
            "rmse_db",
        ],
    )

    write_csv(
        summary_path,
        aggregate,
        [
            "model",
            "condition",
            "maps_evaluated",
            "total_pixels",
            "mae_db_mean",
            "mae_db_std_across_maps",
            "rmse_db_mean",
            "rmse_db_std_across_maps",
        ],
    )

    metadata = {
        "dataset_dir": DATASET_DIR,
        "cache_dir": CACHE_DIR,
        "device": DEVICE,
        "eval_batch_size": (
            EVAL_BATCH_SIZE
        ),
        "dx_meters": DX_METERS,
        "db_min": DB_MIN,
        "db_scale": DB_SCALE,
        "thresholds": thresholds,
        "enabled_models": (
            enabled_specs
        ),
        "metric_protocol": (
            "For each radio map "
            "and each condition, "
            "MAE/RMSE are computed "
            "over valid condition "
            "pixels; final values are "
            "arithmetic means across "
            "valid radio maps."
        ),
        "required_conditions": (
            REQUIRED_CONDITIONS
        ),
        "selection_rule": (
            "Challenger identity must "
            "be fixed from validation "
            "results, not from this "
            "test table."
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print_required_table(
        aggregate
    )

    print("\nSaved:")
    print(
        f"  per-map : "
        f"{per_map_path}"
    )
    print(
        f"  summary : "
        f"{summary_path}"
    )
    print(
        f"  metadata: "
        f"{metadata_path}"
    )

    print("\nSanity check:")
    print(
        "  For the original "
        "Radial {0,1} checkpoint, "
        "ALL_PIXELS_DIAGNOSTIC "
        "should be close to your "
        "ordinary whole-test-set "
        "MAE/RMSE. If not, stop "
        "before using Table-3 results."
    )


if __name__ == "__main__":
    main()