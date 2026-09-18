from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class RSRPSetMapDataset(Dataset):
    """Load per-cell 40x80 maps prepared by prepare_rsrpset_maps.py."""

    def __init__(
        self,
        prepared_root: str | Path,
        split_json: str | Path,
        split: str,
        fraction: float = 1.0,
        subset_seed: int = 0,
        cell_ids: Optional[Sequence[str]] = None,
    ):
        self.prepared_root = Path(prepared_root).resolve()
        self.cell_dir = self.prepared_root / "cells"
        self.split_json = Path(split_json).resolve()
        self.split = split
        self.fraction = float(fraction)
        self.subset_seed = int(subset_seed)

        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(f"fraction must be in (0,1], got {self.fraction}")

        payload = json.loads(self.split_json.read_text(encoding="utf-8"))
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if split not in payload:
            raise KeyError(f"Split file does not contain {split!r}")

        base_ids: List[str] = [str(x) for x in payload[split]]
        if cell_ids is not None:
            requested = [str(x) for x in cell_ids]
            allowed = set(base_ids)
            bad = [x for x in requested if x not in allowed]
            if bad:
                raise ValueError(f"Explicit cell IDs are outside the {split} split: {bad[:10]}")
            ids = requested
        elif split == "train" and self.fraction < 1.0:
            # Nested deterministic subsets: the same shuffle is used for every fraction,
            # so 1% is contained in 5%, which is contained in 10%, etc.
            rng = np.random.default_rng(self.subset_seed)
            perm = np.asarray(base_ids, dtype=object)[rng.permutation(len(base_ids))]
            n = max(1, int(round(len(base_ids) * self.fraction)))
            ids = perm[:n].tolist()
        else:
            ids = base_ids

        self.cell_ids = ids
        missing = [ci for ci in self.cell_ids if not (self.cell_dir / f"{ci}.npz").exists()]
        if missing:
            raise FileNotFoundError(f"Missing prepared cells, first few: {missing[:10]}")

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        ci = self.cell_ids[idx]
        path = self.cell_dir / f"{ci}.npz"
        with np.load(path) as p:
            building = p["building"].astype(np.float32)
            r_map = p["r_map"].astype(np.float32)
            target_norm = p["target_norm"].astype(np.float32)
            gain_db = p["gain_db"].astype(np.float32)
            mask = p["measured_mask"].astype(np.float32)
            rsp_dbm = np.float32(p["rsp_dbm"])

        x = np.stack([building, r_map], axis=0)
        return {
            "input": torch.from_numpy(x),
            "target_norm": torch.from_numpy(target_norm[None, ...]),
            "gain_db": torch.from_numpy(gain_db[None, ...]),
            "mask": torch.from_numpy(mask[None, ...]),
            "rsp_dbm": torch.tensor(rsp_dbm, dtype=torch.float32),
            "ci": ci,
        }

    @property
    def num_cells(self) -> int:
        return len(self.cell_ids)
