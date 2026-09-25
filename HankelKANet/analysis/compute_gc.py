from __future__ import annotations
import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


@dataclass
class Config:
    # ---- 路径 ----
    RUN_DIR: str = "your_training_folder"
    GC_ROOT: Optional[str] = None
    SNAP_DIR: Optional[str] = None
    OUT_CSV: Optional[str] = None
    INDICES_FILE: Optional[str] = None

    # ---- 子集与探针 ----
    GC_N: int = 50
    N_PROBE: int = 50
    BATCH_SIZE: int = 2
    SEED: int = 2025

    # ---- 数据 & 数值 ----
    IMG_SIZE_FALLBACK: int = 256
    DB_MIN_FALLBACK: float = -147.0
    DB_MAX_FALLBACK: float = -47.84
    DATASET_DIR_OVERRIDE: Optional[str] = None  #
    # ---- 运行环境 ----
    DEVICE: Optional[str] = None
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    AMP_FORWARD: bool = False
    EMPTY_CACHE_PER_EPOCH: bool = False

    # ---- 模型构造（通常不需要动） ----
    USE_META_MODEL_KWARGS: bool = False
    BASE_CH_OVERRIDE: Optional[int] = None
    PHYS_CH_OVERRIDE: Optional[int] = None

    # ---- 快照筛选 ----
    EPOCH_FROM: Optional[int] = None
    EPOCH_TO: Optional[int] = None     #
    EVERY: int = 1

CFG = Config()

#grad
# =========================
# 工具函数
# =========================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_epoch_key(p: Path) -> int:
    m = re.search(r"epoch_(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else 10**18


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def ensure_indices(indices_file: Path, seed: int, gc_n: int, val_len: int) -> np.ndarray:
    indices_file.parent.mkdir(parents=True, exist_ok=True)
    if indices_file.exists():
        return np.load(indices_file)

    rng = np.random.default_rng(seed)
    idx = rng.choice(val_len, size=gc_n, replace=False)
    idx = np.sort(idx)
    np.save(indices_file, idx)
    print(f"[OK] Created indices file: {indices_file} (N={gc_n}, seed={seed})")
    return idx


def sample_rademacher(shape: torch.Size, device: torch.device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    u = torch.empty(shape, device=device)
    if generator is None:
        u.bernoulli_(0.5)
    else:
        u.bernoulli_(0.5, generator=generator)
    return u.mul_(2.0).add_(-1.0)


@torch.no_grad()
def sanity_check_output(pred: torch.Tensor) -> None:
    if not torch.isfinite(pred).all():
        raise RuntimeError("Model output has NaN/Inf, cannot compute GC/metrics.")


# =========================
# Dataset / Model
# =========================

def build_val_dataset(img_size: int, dataset_dir: Optional[str] = None):
    from loaders import Dataset_RadioMapSeer
    if dataset_dir is None:
        return Dataset_RadioMapSeer(phase="val", img_size=img_size)
    return Dataset_RadioMapSeer(phase="val", img_size=img_size, dir_dataset=dataset_dir)


def build_model(meta: Dict[str, Any], cfg: Config) -> torch.nn.Module:
    from modules import HKANNet

    kwargs: Dict[str, Any] = {}
    if cfg.USE_META_MODEL_KWARGS and isinstance(meta.get("model_kwargs", None), dict):
        kwargs.update(meta["model_kwargs"])

    if cfg.BASE_CH_OVERRIDE is not None:
        kwargs["base_ch"] = int(cfg.BASE_CH_OVERRIDE)
    if cfg.PHYS_CH_OVERRIDE is not None:
        kwargs["phys_ch"] = int(cfg.PHYS_CH_OVERRIDE)

    model = HKANNet(**kwargs) if kwargs else HKANNet()
    print(f"[Model] HKANNet kwargs used: {kwargs if kwargs else '{default}'}")
    return model


# =========================
# Label stats for R2
# =========================

def precompute_label_stats_norm(dl: DataLoader, device: torch.device) -> Tuple[float, float, int]:
    sum_y = 0.0
    sum_y2 = 0.0
    n = 0
    for inputs, label in dl:
        label = label.to(device=device, dtype=torch.float32)
        sum_y += float(label.sum().item())
        sum_y2 += float((label * label).sum().item())
        n += label.numel()

    mean = sum_y / max(n, 1)
    sst = sum_y2 - n * (mean ** 2)
    sst = max(sst, 1e-12)
    return mean, sst, n


def precompute_label_stats_db(dl: DataLoader, db_min: float, db_max: float, device: torch.device) -> Tuple[float, float, int]:
    scale = db_max - db_min
    sum_y = 0.0
    sum_y2 = 0.0
    n = 0
    for inputs, label in dl:
        label = label.to(device=device, dtype=torch.float32)
        label_db = label * scale + db_min
        sum_y += float(label_db.sum().item())
        sum_y2 += float((label_db * label_db).sum().item())
        n += label_db.numel()

    mean = sum_y / max(n, 1)
    sst = sum_y2 - n * (mean ** 2)
    sst = max(sst, 1e-12)
    return mean, sst, n


# =========================
# Core computation
# =========================

def compute_epoch_gc_and_metrics(
    model: torch.nn.Module,
    dl: DataLoader,
    db_min: float,
    db_max: float,
    n_probe: int,
    seed: int,
    device: torch.device,
    label_sst_norm: float,
    label_n_elem_norm: int,
    label_sst_db: float,
    label_n_elem_db: int,
    amp_forward: bool = False,
) -> Tuple[float, float, float, float, float, float, float]:
    scale = db_max - db_min

    gc_sum = 0.0
    n_samples = 0

    sse_norm = 0.0
    sae_norm = 0.0

    sse_db = 0.0
    sae_db = 0.0

    for p in model.parameters():
        p.requires_grad_(False)

    model.eval()

    for batch_id, (inputs, label) in enumerate(dl):
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        label = label.to(device=device, dtype=torch.float32, non_blocking=True)
        inputs.requires_grad_(True)

        with torch.enable_grad():
            if amp_forward and device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(inputs)
            else:
                pred = model(inputs)

        sanity_check_output(pred)

        # ---- metrics（detach 省显存）----
        pred_det = pred.detach()

        diff_norm = pred_det - label
        sse_norm += float((diff_norm * diff_norm).sum().item())
        sae_norm += float(diff_norm.abs().sum().item())

        pred_db = pred_det * scale + db_min
        label_db = label * scale + db_min
        diff_db = pred_db - label_db
        sse_db += float((diff_db * diff_db).sum().item())
        sae_db += float(diff_db.abs().sum().item())

        # ---- GC：Hutchinson ----
        B = inputs.shape[0]
        gc_batch = torch.zeros((B,), device=device, dtype=torch.float32)

        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed + batch_id * 10007))

        for t in range(n_probe):
            v = sample_rademacher(pred.shape, device=device, generator=gen)
            s = (pred * v).sum()
            grad = torch.autograd.grad(
                outputs=s,
                inputs=inputs,
                retain_graph=(t < n_probe - 1),
                create_graph=False,

            )[0]
            gc_batch += (grad.reshape(B, -1) ** 2).sum(dim=1)
            del grad, v, s

        gc_batch /= float(n_probe)
        gc_sum += float(gc_batch.sum().item())
        n_samples += B

        del pred, pred_det, diff_norm, pred_db, label_db, diff_db, gc_batch, inputs, label

    val_mae = sae_norm / max(label_n_elem_norm, 1)
    val_rmse = math.sqrt(sse_norm / max(label_n_elem_norm, 1))
    val_r2 = 1.0 - (sse_norm / max(label_sst_norm, 1e-12))

    val_mae_db = sae_db / max(label_n_elem_db, 1)
    val_rmse_db = math.sqrt(sse_db / max(label_n_elem_db, 1))
    val_r2_db = 1.0 - (sse_db / max(label_sst_db, 1e-12))

    GCsum  = gc_sum / max(n_samples, 1)
    GCavg  = GCsum / (2*256*256)
    return GCsum, GCavg, val_mae, val_rmse, val_r2, val_mae_db, val_rmse_db, val_r2_db



# =========================
# Main
# =========================

def main(cfg: Config) -> None:
    set_seed(cfg.SEED)

    run_dir = Path(cfg.RUN_DIR)
    gc_root = Path(cfg.GC_ROOT) if cfg.GC_ROOT else (run_dir / "gc_artifacts")
    snap_dir = Path(cfg.SNAP_DIR) if cfg.SNAP_DIR else (gc_root / "snapshots")
    meta = load_json(gc_root / "meta.json")

    # meta overrides（优先 meta）
    img_size = int(meta.get("img_size", cfg.IMG_SIZE_FALLBACK))
    db_min = float(meta.get("db_min", cfg.DB_MIN_FALLBACK))
    db_max = float(meta.get("db_max", cfg.DB_MAX_FALLBACK))

    dataset_dir = cfg.DATASET_DIR_OVERRIDE
    if dataset_dir is None:
        dataset_dir = meta.get("dataset_dir", None)

    # device
    if cfg.DEVICE is not None:
        device = torch.device(cfg.DEVICE)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not snap_dir.exists():
        raise FileNotFoundError(f"Snapshots dir not found: {snap_dir}")

    indices_file = Path(cfg.INDICES_FILE) if cfg.INDICES_FILE else (gc_root / f"eval_indices_val{cfg.GC_N}.npy")
    out_csv = Path(cfg.OUT_CSV) if cfg.OUT_CSV else (gc_root / f"gc_metrics_hkan_val{cfg.GC_N}_probe{cfg.N_PROBE}.csv")

    # dataset subset
    val_set = build_val_dataset(img_size=img_size, dataset_dir=dataset_dir)
    idx = ensure_indices(indices_file, seed=cfg.SEED, gc_n=cfg.GC_N, val_len=len(val_set))
    subset = Subset(val_set, idx.tolist())

    dl = DataLoader(
        subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(cfg.PIN_MEMORY and device.type == "cuda"),
        drop_last=False,
    )

    # label stats（固定 across epochs）
    mean_norm, sst_norm, n_elem_norm = precompute_label_stats_norm(dl, device=device)
    mean_db, sst_db, n_elem_db = precompute_label_stats_db(dl, db_min=db_min, db_max=db_max, device=device)
    print(f"[OK] Label stats (norm): mean={mean_norm:.6f}, SST={sst_norm:.3e}, n_elem={n_elem_norm}")
    print(f"[OK] Label stats (dB):   mean_db={mean_db:.4f}, SST={sst_db:.3e}, n_elem={n_elem_db}")

    # snapshots list + filter
    snaps_all = sorted(list(snap_dir.glob("epoch_*.pt")), key=natural_epoch_key)
    if not snaps_all:
        raise FileNotFoundError(f"No snapshot files found in: {snap_dir}")

    snaps: List[Path] = []
    for p in snaps_all:
        e = natural_epoch_key(p)
        if cfg.EPOCH_FROM is not None and e < cfg.EPOCH_FROM:
            continue
        if cfg.EPOCH_TO is not None and e > cfg.EPOCH_TO:
            continue
        if cfg.EVERY > 1 and (e % cfg.EVERY != 0):
            continue
        snaps.append(p)

    if not snaps:
        raise RuntimeError("No snapshots left after filtering. Check EPOCH_FROM/EPOCH_TO/EVERY in CONFIG.")

    # model
    model = build_model(meta, cfg).to(device=device)

    rows: List[Dict[str, Any]] = []

    for p in snaps:
        pack = torch.load(p, map_location="cpu")
        epoch = int(pack.get("epoch", -1))
        step = int(pack.get("step", -1))

        state = pack.get("model", None) or pack.get("model_state_dict", None)
        if state is None:
            raise KeyError(f"Snapshot {p} does not contain 'model' or 'model_state_dict'.")

        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"load_state_dict failed for {p.name}.\n"
                f"通常是 HKANNet 构造参数与训练时不一致。\n"
                f"你可以在 CONFIG 里设置：\n"
                f"  - USE_META_MODEL_KWARGS=True（若 meta.json 里 model_kwargs 正确）\n"
                f"  - 或 BASE_CH_OVERRIDE / PHYS_CH_OVERRIDE 强行指定\n"
                f"原始错误：{e}"
            ) from e

        epoch_seed = int(cfg.SEED)  # 固定跨 epoch，最大化可比性

        GCsum, GCavg, val_mae, val_rmse, val_r2, val_mae_db, val_rmse_db, val_r2_db = compute_epoch_gc_and_metrics(
            model=model,
            dl=dl,
            db_min=db_min,
            db_max=db_max,
            n_probe=cfg.N_PROBE,
            seed=epoch_seed,
            device=device,
            label_sst_norm=sst_norm,
            label_n_elem_norm=n_elem_norm,
            label_sst_db=sst_db,
            label_n_elem_db=n_elem_db,
            amp_forward=cfg.AMP_FORWARD,
        )

        rows.append(dict(
            epoch=epoch,
            step=step,
            GCsum=float(GCsum),
            GCavg=float(GCavg),
            val_mae=float(val_mae),
            val_rmse=float(val_rmse),
            val_r2=float(val_r2),
            val_mae_db=float(val_mae_db),
            val_rmse_db=float(val_rmse_db),
            val_r2_db=float(val_r2_db),
        ))

        print(
            f"[Epoch {epoch:04d}] "
            f"GCsum={GCsum:.4e} | "
            f"GCavg={GCavg:.4e} | "
            f"MAE={val_mae:.5f} RMSE={val_rmse:.5f} R2={val_r2:.4f} | "
            f"MAE(dB)={val_mae_db:.3f} RMSE(dB)={val_rmse_db:.3f} R2(dB)={val_r2_db:.4f}"
        )

        if cfg.EMPTY_CACHE_PER_EPOCH and device.type == "cuda":
            torch.cuda.empty_cache()

    # write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "step", "GCsum", "GCavg", "val_mae", "val_rmse", "val_r2", "val_mae_db", "val_rmse_db", "val_r2_db"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # settings dump（把 CONFIG 和 meta 关键字段一起存下来）
    settings = {
        "config": asdict(cfg),
        "resolved": {
            "gc_root": str(gc_root),
            "snap_dir": str(snap_dir),
            "indices_file": str(indices_file),
            "out_csv": str(out_csv),
            "img_size": img_size,
            "db_min": db_min,
            "db_max": db_max,
            "dataset_dir": dataset_dir,
            "device": str(device),
        },
        "meta_excerpt": {
            "img_size": meta.get("img_size", None),
            "db_min": meta.get("db_min", None),
            "db_max": meta.get("db_max", None),
            "dataset_dir": meta.get("dataset_dir", None),
            "model_kwargs": meta.get("model_kwargs", None),
        }
    }
    settings_path = out_csv.with_suffix(".settings.json")
    settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[Done] Wrote: {out_csv}")
    print(f"[Done] Settings: {settings_path}")


if __name__ == "__main__":
    main(CFG)
