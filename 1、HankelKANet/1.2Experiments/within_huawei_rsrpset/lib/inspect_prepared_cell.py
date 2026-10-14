from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--ci", required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def save_map(arr, title, path, mask=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=(5, 3.5))
    show = np.asarray(arr).copy()
    if mask is not None:
        show = np.where(mask, show, np.nan)
    plt.imshow(show, aspect="auto", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    path = args.prepared_root.resolve() / "cells" / f"{args.ci}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    out = args.output or (args.prepared_root.resolve() / "diagnostics" / str(args.ci))
    out.mkdir(parents=True, exist_ok=True)

    with np.load(path) as p:
        data = {k: p[k] for k in p.files}

    mask = data["measured_mask"].astype(bool)
    env_mask = data.get("environment_observed_mask", np.zeros_like(mask)).astype(bool)
    building = data["building"].astype(float)

    print(f"CI={args.ci}")
    for k, v in data.items():
        if np.ndim(v) == 0:
            print(f"{k:32s}: {v.item()}")
        else:
            finite = np.asarray(v)[np.isfinite(v)]
            print(
                f"{k:32s}: shape={v.shape}, dtype={v.dtype}, "
                f"min={finite.min() if finite.size else np.nan:.5g}, "
                f"max={finite.max() if finite.size else np.nan:.5g}"
            )

    print(f"Measured grid coverage:          {mask.mean():.2%}")
    print(f"Environment observed coverage:   {env_mask.mean():.2%}")
    print(f"Completed building fraction:     {building.mean():.2%}")

    if "building_observed" in data:
        obs_b = data["building_observed"].astype(bool)
        obs_frac = (obs_b & env_mask).sum() / max(int(env_mask.sum()), 1)
        print(f"Observed building fraction:      {obs_frac:.2%}")
    if "target_on_building_mask" in data:
        conflict = data["target_on_building_mask"].astype(bool)
        print(
            "Target-on-building conflict:     "
            f"{conflict.sum() / max(int(mask.sum()), 1):.2%}"
        )
    if "environment_label_conflict_mask" in data:
        c = data["environment_label_conflict_mask"].astype(bool)
        print(
            "Observed-label grid conflicts:   "
            f"{c.sum() / max(int(env_mask.sum()), 1):.2%}"
        )

    save_map(building, "Reconstructed building map", out / "building.png", vmin=0, vmax=1)
    save_map(data["r_map"], "Normalized radial distance rho", out / "r_map.png", vmin=0, vmax=1)
    save_map(
        data["gain_db"],
        "Measured gain (dB), measured grids only",
        out / "gain_db.png",
        mask,
    )
    save_map(
        data["rsrp_dbm"],
        "Measured RSRP (dBm), measured grids only",
        out / "rsrp_dbm.png",
        mask,
    )
    save_map(mask.astype(float), "Measured-target mask", out / "measured_mask.png", vmin=0, vmax=1)

    if "building_observed" in data:
        save_map(
            data["building_observed"].astype(float),
            "Observed building labels only",
            out / "building_observed.png",
            env_mask,
            vmin=0,
            vmax=1,
        )
    save_map(
        env_mask.astype(float),
        "Environment-observed mask",
        out / "environment_observed_mask.png",
        vmin=0,
        vmax=1,
    )
    if "target_on_building_mask" in data:
        save_map(
            data["target_on_building_mask"].astype(float),
            "Measured targets falling on completed building grids",
            out / "target_on_building_mask.png",
            vmin=0,
            vmax=1,
        )

    print(f"Saved diagnostics to {out}")


if __name__ == "__main__":
    main()
