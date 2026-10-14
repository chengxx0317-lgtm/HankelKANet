import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stratified_common import (
    NUM_TX,
    SPLIT_SEED,
    building_density,
    compute_distance_rmap_phi,
    compute_los_mask_vectorized,
    get_scene_split,
    load_scene_building,
    load_tx_mask,
    los_scene_cache_path,
    pack_bool_mask,
    save_scene_los_cache,
)


# ============================================================================
# PyCharm configuration: edit only this block
# ============================================================================
DATASET_DIR = r"D:\PycharmProjects\PythonProjects\CXX\PEFNet\data\RadioMapSeer"
CACHE_DIR = r"./challenging_condition_cache"

IMG_SIZE_USED = 256
DX_METERS = 1.0   # aligned with your original loaders.py

# Pre-fixed rules.
SHORT_DISTANCE_QUANTILE = 0.25
LONG_DISTANCE_QUANTILE = 0.75
HIGH_DENSITY_QUANTILE = 0.75

# Deterministic TRAIN-split distance sampling:
# 500 scenes x 80 Tx x 32 outdoor Rx ~= 1.28 million samples.
DISTANCE_SAMPLES_PER_TX = 32
THRESHOLD_RANDOM_SEED = 20260816

# LOS cache generation.
LOS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOS_RAY_BATCH_SIZE = 16384

FORCE_RECOMPUTE_THRESHOLDS = False
FORCE_RECOMPUTE_LOS = False

SAVE_VISUAL_CHECKS = True
VISUAL_CHECK_SCENES = 3
VISUAL_CHECK_TX_ID = 0


def compute_training_thresholds():
    train_scenes = get_scene_split()["train"]
    rng = np.random.default_rng(THRESHOLD_RANDOM_SEED)

    densities = []
    sampled_distances = []

    print("=" * 90)
    print("Computing thresholds from TRAIN split only")
    print(f"Train scenes             : {len(train_scenes)}")
    print(f"Distance samples per Tx  : {DISTANCE_SAMPLES_PER_TX}")
    print(f"Expected distance samples: ~{len(train_scenes)*NUM_TX*DISTANCE_SAMPLES_PER_TX:,}")
    print("=" * 90)

    t0 = time.time()

    for scene_pos, scene_id in enumerate(train_scenes, 1):
        scene_id = int(scene_id)

        _, bld_mask = load_scene_building(
            DATASET_DIR, scene_id, IMG_SIZE_USED
        )
        densities.append(building_density(bld_mask))

        outdoor_y, outdoor_x = np.where(~bld_mask)
        if outdoor_y.size == 0:
            raise RuntimeError(
                f"Scene {scene_id} has no outdoor pixels."
            )

        for tx_id in range(NUM_TX):
            tx = load_tx_mask(
                DATASET_DIR, scene_id, tx_id, IMG_SIZE_USED
            )
            _, _, _, (y0, x0) = compute_distance_rmap_phi(
                tx, dx=DX_METERS
            )

            k = min(DISTANCE_SAMPLES_PER_TX, outdoor_y.size)
            choose = rng.choice(outdoor_y.size, size=k, replace=False)

            dy = outdoor_y[choose].astype(np.float32) - float(y0)
            dxv = outdoor_x[choose].astype(np.float32) - float(x0)
            d = np.sqrt(dxv * dxv + dy * dy) * float(DX_METERS)

            # Tx itself is not a receiver in this stratified test.
            d = d[d > 0.0]
            sampled_distances.append(d.astype(np.float32))

        if scene_pos % 25 == 0 or scene_pos == len(train_scenes):
            print(
                f"[Thresholds] {scene_pos:3d}/{len(train_scenes)} scenes | "
                f"elapsed={time.time()-t0:.1f}s"
            )

    distances = np.concatenate(sampled_distances, axis=0)
    densities = np.asarray(densities, dtype=np.float32)

    short_thr = float(
        np.quantile(distances, SHORT_DISTANCE_QUANTILE)
    )
    long_thr = float(
        np.quantile(distances, LONG_DISTANCE_QUANTILE)
    )
    high_density_thr = float(
        np.quantile(densities, HIGH_DENSITY_QUANTILE)
    )

    return {
        "dataset": "RadioMapSeer",
        "split_seed": SPLIT_SEED,
        "img_size": IMG_SIZE_USED,
        "dx_meters": DX_METERS,

        "distance_threshold_source": (
            "TRAIN-split outdoor Tx-Rx samples only"
        ),
        "distance_sampling_seed": THRESHOLD_RANDOM_SEED,
        "distance_samples_per_tx": DISTANCE_SAMPLES_PER_TX,
        "distance_sample_count": int(distances.size),
        "short_distance_quantile": SHORT_DISTANCE_QUANTILE,
        "long_distance_quantile": LONG_DISTANCE_QUANTILE,
        "short_distance_max_m": short_thr,
        "long_distance_min_m": long_thr,

        "building_density_definition": (
            "number_of_building_pixels / (H*W)"
        ),
        "building_density_threshold_source": (
            "all TRAIN-split scene building maps"
        ),
        "training_scene_count_for_density": int(densities.size),
        "high_density_quantile": HIGH_DENSITY_QUANTILE,
        "high_building_density_min": high_density_thr,

        "evaluation_receiver_policy": (
            "Stratified metrics use outdoor Rx pixels only and exclude Tx."
        ),
        "los_definition": (
            "Tx-to-Rx grid centerline does not traverse a building pixel "
            "before the outdoor Rx endpoint."
        ),
        "los_implementation": (
            "vectorized DDA centerline test; source/Rx endpoints excluded"
        ),
    }


def save_visual_check(
    cache_dir, scene_id, tx_id, bld_mask, los_mask, tx_coord
):
    out_dir = Path(cache_dir) / "visual_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    y0, x0 = tx_coord
    outdoor = ~bld_mask
    nlos = outdoor & (~los_mask)
    nlos[y0, x0] = False

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(bld_mask, cmap="gray")
    axes[0].scatter([x0], [y0], s=18, marker="x")
    axes[0].set_title(
        f"Building | scene={scene_id}, tx={tx_id}"
    )

    axes[1].imshow(los_mask, cmap="gray")
    axes[1].scatter([x0], [y0], s=18, marker="x")
    axes[1].set_title("LOS outdoor pixels")

    axes[2].imshow(nlos, cmap="gray")
    axes[2].scatter([x0], [y0], s=18, marker="x")
    axes[2].set_title("NLOS outdoor pixels")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        out_dir / (
            f"scene_{int(scene_id):03d}_tx_{int(tx_id):02d}.png"
        ),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def prepare_test_los_cache():
    test_scenes = get_scene_split()["test"]

    cache_root = Path(CACHE_DIR)
    (cache_root / "los_masks").mkdir(
        parents=True, exist_ok=True
    )

    print("\n" + "=" * 90)
    print("Preparing TEST-set LOS masks")
    print(f"LOS device : {LOS_DEVICE}")
    print(f"Ray batch  : {LOS_RAY_BATCH_SIZE}")
    print(f"Test scenes: {len(test_scenes)} x {NUM_TX} Tx")
    print("=" * 90)

    t_all = time.time()

    for scene_pos, scene_id in enumerate(test_scenes, 1):
        scene_id = int(scene_id)
        scene_cache = los_scene_cache_path(
            CACHE_DIR, scene_id
        )

        if scene_cache.exists() and not FORCE_RECOMPUTE_LOS:
            try:
                data = np.load(
                    scene_cache, allow_pickle=False
                )
                ok = (
                    data["los_packed"].shape[0] == NUM_TX
                    and tuple(data["shape"].tolist())
                    == (IMG_SIZE_USED, IMG_SIZE_USED)
                )
                data.close()
                if ok:
                    print(
                        f"[LOS] scene {scene_id:03d} "
                        f"({scene_pos:3d}/{len(test_scenes)}) cached -> skip"
                    )
                    continue
            except Exception:
                pass

        _, bld_mask = load_scene_building(
            DATASET_DIR, scene_id, IMG_SIZE_USED
        )

        packed_masks = []
        tx_coords = []
        t_scene = time.time()

        for tx_id in range(NUM_TX):
            tx = load_tx_mask(
                DATASET_DIR, scene_id, tx_id, IMG_SIZE_USED
            )
            _, _, _, tx_coord = compute_distance_rmap_phi(
                tx, dx=DX_METERS
            )

            los = compute_los_mask_vectorized(
                bld_mask,
                tx_coord,
                device=LOS_DEVICE,
                ray_batch_size=LOS_RAY_BATCH_SIZE,
            )

            packed_masks.append(pack_bool_mask(los))
            tx_coords.append(tx_coord)

            if (
                SAVE_VISUAL_CHECKS
                and scene_pos <= VISUAL_CHECK_SCENES
                and tx_id == VISUAL_CHECK_TX_ID
            ):
                save_visual_check(
                    CACHE_DIR,
                    scene_id,
                    tx_id,
                    bld_mask,
                    los,
                    tx_coord,
                )

            if (tx_id + 1) % 10 == 0:
                print(
                    f"  scene {scene_id:03d}: "
                    f"Tx {tx_id+1:02d}/{NUM_TX} | "
                    f"scene_elapsed={time.time()-t_scene:.1f}s"
                )

        save_scene_los_cache(
            CACHE_DIR,
            scene_id,
            packed_masks=np.stack(
                packed_masks, axis=0
            ),
            tx_coords=np.asarray(tx_coords),
            shape=(IMG_SIZE_USED, IMG_SIZE_USED),
        )

        print(
            f"[LOS] scene {scene_id:03d} complete "
            f"({scene_pos:3d}/{len(test_scenes)}) | "
            f"scene_time={time.time()-t_scene:.1f}s | "
            f"total={time.time()-t_all:.1f}s"
        )


def main():
    cache_root = Path(CACHE_DIR)
    cache_root.mkdir(parents=True, exist_ok=True)
    threshold_path = cache_root / "thresholds.json"

    if (
        threshold_path.exists()
        and not FORCE_RECOMPUTE_THRESHOLDS
    ):
        thresholds = json.loads(
            threshold_path.read_text(encoding="utf-8")
        )
        print(
            f"Using existing threshold file: {threshold_path}"
        )
    else:
        thresholds = compute_training_thresholds()
        threshold_path.write_text(
            json.dumps(
                thresholds,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8",
        )
        print(f"\nThresholds saved: {threshold_path}")

    print("\nFixed TRAIN-derived thresholds:")
    print(
        f"  Short distance : "
        f"d <= {thresholds['short_distance_max_m']:.4f} m "
        f"(Q{int(100*thresholds['short_distance_quantile'])})"
    )
    print(
        f"  Long distance  : "
        f"d >= {thresholds['long_distance_min_m']:.4f} m "
        f"(Q{int(100*thresholds['long_distance_quantile'])})"
    )
    print(
        f"  High density   : "
        f"density >= {thresholds['high_building_density_min']:.6f} "
        f"(Q{int(100*thresholds['high_density_quantile'])})"
    )
    print(
        f"  Distance samples used: "
        f"{thresholds['distance_sample_count']:,}"
    )

    prepare_test_los_cache()

    print("\n" + "=" * 90)
    print("PREPARATION FINISHED")
    print("Next: run evaluate_challenging_conditions.py")
    print("=" * 90)


if __name__ == "__main__":
    main()