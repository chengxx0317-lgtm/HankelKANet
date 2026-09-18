import json
from pathlib import Path

import numpy as np
import torch
from skimage import io, transform
from torch.utils.data import Dataset

NUM_SCENES = 700
NUM_TX = 80
IMG_SIZE = 256
SPLIT_SEED = 42

DB_MIN = -147.0
DB_MAX = -47.84
DB_SCALE = DB_MAX - DB_MIN


def get_scene_split():
    all_inds = np.arange(0, NUM_SCENES)
    np.random.seed(SPLIT_SEED)
    np.random.shuffle(all_inds)
    return {
        "train": all_inds[0:500].copy(),
        "val": all_inds[500:600].copy(),
        "test": all_inds[600:700].copy(),
    }


def normalize_image01(arr):
    arr = np.asarray(arr).astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def load_scene_building(dataset_dir, scene_id, img_size=IMG_SIZE):
    path = Path(dataset_dir) / "png" / "buildings_complete" / f"{scene_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Building map not found: {path}")
    bld = normalize_image01(io.imread(path))
    if bld.shape[0] != img_size or bld.shape[1] != img_size:
        bld = transform.resize(
            bld, (img_size, img_size),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.float32)
    bld_mask = bld > 0.5
    return bld.astype(np.float32), bld_mask


def load_tx_mask(dataset_dir, scene_id, tx_id, img_size=IMG_SIZE):
    path = Path(dataset_dir) / "png" / "antennas" / f"{scene_id}_{tx_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Tx map not found: {path}")
    tx = normalize_image01(io.imread(path))
    if tx.shape[0] != img_size or tx.shape[1] != img_size:
        tx = transform.resize(
            tx, (img_size, img_size),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.float32)
    return tx.astype(np.float32)


def load_gain(dataset_dir, scene_id, tx_id, img_size=IMG_SIZE):
    path = Path(dataset_dir) / "gain" / "DPM" / f"{scene_id}_{tx_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"DPM gain map not found: {path}")
    gain = normalize_image01(io.imread(path))
    if gain.ndim == 3:
        gain = gain[..., 0]
    if gain.shape[0] != img_size or gain.shape[1] != img_size:
        gain = transform.resize(
            gain, (img_size, img_size),
            order=1, preserve_range=True, anti_aliasing=False
        ).astype(np.float32)
    return gain.astype(np.float32)


def get_tx_coord(tx_mask):
    y0, x0 = np.where(tx_mask > 0)
    if len(x0) == 0:
        h, w = tx_mask.shape
        return h // 2, w // 2
    return int(y0[0]), int(x0[0])


def compute_distance_rmap_phi(tx_mask, dx=1.0):
    h, w = tx_mask.shape
    y0, x0 = get_tx_coord(tx_mask)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    distance_m = (
        np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2).astype(np.float32)
        * float(dx)
    )

    # Match the original loader: +1e-8 then per-sample min-max normalization.
    r = distance_m + 1e-8
    eps = 1e-9
    r_map = (r - r.min()) / (r.max() - r.min() + eps)
    r_map = r_map.astype(np.float32)

    phi_map = np.arctan2(
        (yy - y0).astype(np.float32),
        (xx - x0).astype(np.float32),
    ).astype(np.float32)
    phi_map[y0, x0] = 0.0

    return distance_m.astype(np.float32), r_map, phi_map, (y0, x0)


def building_density(building_mask):
    return float(np.mean(np.asarray(building_mask, dtype=np.float32)))


def pack_bool_mask(mask):
    return np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1))


def unpack_bool_mask(packed, shape):
    total = int(np.prod(shape))
    flat = np.unpackbits(np.asarray(packed, dtype=np.uint8), count=total)
    return flat.reshape(shape).astype(np.bool_)


def compute_los_mask_vectorized(
    building_mask,
    tx_coord,
    device="cuda",
    ray_batch_size=16384,
):

    building_mask = np.asarray(building_mask, dtype=np.bool_)
    h, w = building_mask.shape
    y0, x0 = int(tx_coord[0]), int(tx_coord[1])

    outdoor = ~building_mask
    outdoor[y0, x0] = False
    rx_y, rx_x = np.where(outdoor)

    los_flat = np.zeros(rx_y.shape[0], dtype=np.bool_)
    if rx_y.size == 0:
        return np.zeros((h, w), dtype=np.bool_)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    bld_t = torch.from_numpy(building_mask).to(dev)

    n = rx_y.shape[0]
    for start in range(0, n, int(ray_batch_size)):
        end = min(start + int(ray_batch_size), n)

        y1 = torch.from_numpy(rx_y[start:end].astype(np.int64)).to(dev)
        x1 = torch.from_numpy(rx_x[start:end].astype(np.int64)).to(dev)

        dy = y1 - y0
        dxv = x1 - x0
        steps = torch.maximum(torch.abs(dy), torch.abs(dxv)).to(torch.int64)

        is_los = steps <= 1

        active_idx = torch.where(steps > 1)[0]
        if active_idx.numel() > 0:
            dy_a = dy[active_idx].to(torch.float32)
            dx_a = dxv[active_idx].to(torch.float32)
            steps_a = steps[active_idx]

            max_steps = int(steps_a.max().item())
            t = torch.arange(
                1, max_steps, device=dev, dtype=torch.float32
            ).view(1, -1)

            valid = t < steps_a.to(torch.float32).view(-1, 1)
            denom = steps_a.to(torch.float32).view(-1, 1)

            y_line = torch.round(
                float(y0) + dy_a.view(-1, 1) * (t / denom)
            ).to(torch.long)
            x_line = torch.round(
                float(x0) + dx_a.view(-1, 1) * (t / denom)
            ).to(torch.long)

            y_line.clamp_(0, h - 1)
            x_line.clamp_(0, w - 1)

            blocked = bld_t[y_line, x_line] & valid
            has_block = torch.any(blocked, dim=1)
            is_los[active_idx] = ~has_block

        los_flat[start:end] = is_los.detach().cpu().numpy()

    los_mask = np.zeros((h, w), dtype=np.bool_)
    los_mask[rx_y, rx_x] = los_flat
    los_mask[y0, x0] = False
    return los_mask


def los_scene_cache_path(cache_dir, scene_id):
    return Path(cache_dir) / "los_masks" / f"scene_{int(scene_id):03d}.npz"


def save_scene_los_cache(cache_dir, scene_id, packed_masks, tx_coords, shape):
    path = los_scene_cache_path(cache_dir, scene_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        los_packed=np.asarray(packed_masks, dtype=np.uint8),
        tx_coords=np.asarray(tx_coords, dtype=np.int16),
        shape=np.asarray(shape, dtype=np.int16),
    )
    return path


def load_scene_los_cache(cache_dir, scene_id):
    path = los_scene_cache_path(cache_dir, scene_id)
    if not path.exists():
        raise FileNotFoundError(
            f"LOS cache missing: {path}\n"
            f"Run prepare_challenging_conditions.py first."
        )
    with np.load(path, allow_pickle=False) as data:
        los_packed = data["los_packed"].copy()
        tx_coords = data["tx_coords"].copy()
        shape = tuple(int(v) for v in data["shape"].tolist())
    return {
        "los_packed": los_packed,
        "tx_coords": tx_coords,
        "shape": shape,
    }


class StratifiedRadioMapSeerDataset(Dataset):
    def __init__(self, dataset_dir, phase="test", img_size=IMG_SIZE, dx=1.0):
        self.dataset_dir = str(dataset_dir)
        self.phase = str(phase)
        self.img_size = int(img_size)
        self.dx = float(dx)
        self.map_inds = get_scene_split()[self.phase]
        self.numTx = NUM_TX

    def __len__(self):
        return len(self.map_inds) * self.numTx

    def __getitem__(self, idx):
        idxr = idx // self.numTx
        tx_id = idx % self.numTx
        scene_id = int(self.map_inds[idxr])

        bld, bld_mask = load_scene_building(
            self.dataset_dir, scene_id, self.img_size
        )
        tx = load_tx_mask(
            self.dataset_dir, scene_id, tx_id, self.img_size
        )
        gain = load_gain(
            self.dataset_dir, scene_id, tx_id, self.img_size
        )

        distance_m, r_map, phi_map, tx_coord = compute_distance_rmap_phi(
            tx, dx=self.dx
        )

        return {
            "building": torch.from_numpy(bld).unsqueeze(0).float(),
            "building_mask": torch.from_numpy(
                bld_mask.astype(np.uint8)
            ).unsqueeze(0),
            "r_map": torch.from_numpy(r_map).unsqueeze(0).float(),
            "phi_map": torch.from_numpy(phi_map).unsqueeze(0).float(),
            "label": torch.from_numpy(gain).unsqueeze(0).float(),
            "distance_m": torch.from_numpy(distance_m).unsqueeze(0).float(),
            "scene_id": scene_id,
            "tx_id": tx_id,
            "tx_coord": torch.tensor(tx_coord, dtype=torch.int16),
            "sample_id": f"{scene_id}_{tx_id}",
        }


def load_thresholds(cache_dir):
    path = Path(cache_dir) / "thresholds.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Threshold file missing: {path}\n"
            f"Run prepare_challenging_conditions.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))