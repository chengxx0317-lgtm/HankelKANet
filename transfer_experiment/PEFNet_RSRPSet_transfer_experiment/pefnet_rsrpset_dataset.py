from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from scipy import special
from torch.utils.data import Dataset


def compute_einc(
    tx_mask: np.ndarray,
    carrier_frequency_hz: float = 2.6e9,
    dx_m: float = 5.0,
    normalize: bool = True,
):
    c = 3e8
    k = 2.0 * np.pi * float(carrier_frequency_hz) / c

    h, w = tx_mask.shape
    y0, x0 = np.where(tx_mask > 0)
    if len(x0) == 0:
        raise ValueError("Tx mask contains no transmitter pixel.")
    y0, x0 = int(y0[0]), int(x0[0])

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2) * float(dx_m) + 1e-8

    h0 = special.hankel2(0, k * r)
    e_inc = -1j / 4.0 * h0

    real = np.real(e_inc).astype(np.float32)
    imag = np.imag(e_inc).astype(np.float32)
    real_raw = real.copy()
    imag_raw = imag.copy()

    if normalize:
        eps = 1e-9
        real = (real - real.min()) / (real.max() - real.min() + eps)
        imag = (imag - imag.min()) / (imag.max() - imag.min() + eps)

    return real, imag, real_raw, imag_raw


class PEFNetRSRPSetDataset(Dataset):
    def __init__(
        self,
        prepared_root: str | Path,
        split_json: str | Path,
        split: str,
        carrier_frequency_hz: float = 2.6e9,
        dx_m: float = 5.0,
        cell_ids: Optional[Sequence[str]] = None,
        tx_row: int = 0,
        tx_col: int = 0,
    ):
        self.prepared_root = Path(prepared_root).resolve()
        self.cell_dir = self.prepared_root / "cells"
        self.split_json = Path(split_json).resolve()
        self.split = split
        self.carrier_frequency_hz = float(carrier_frequency_hz)
        self.dx_m = float(dx_m)
        self.tx_row = int(tx_row)
        self.tx_col = int(tx_col)

        payload = json.loads(self.split_json.read_text(encoding="utf-8"))
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if split not in payload:
            raise KeyError(f"Split file does not contain {split!r}")

        base_ids = [str(x) for x in payload[split]]
        if cell_ids is None:
            self.cell_ids = base_ids
        else:
            requested = [str(x) for x in cell_ids]
            allowed = set(base_ids)
            bad = [x for x in requested if x not in allowed]
            if bad:
                raise ValueError(
                    f"Explicit cell IDs are outside the {split} split: {bad[:10]}"
                )
            self.cell_ids = requested

        if not self.cell_ids:
            raise RuntimeError(f"No cells in split={split}.")

        missing = [
            ci for ci in self.cell_ids
            if not (self.cell_dir / f"{ci}.npz").exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing prepared cells, first few: {missing[:10]}")

        with np.load(self.cell_dir / f"{self.cell_ids[0]}.npz") as pack:
            h, w = pack["building"].shape

        if not (0 <= self.tx_row < h and 0 <= self.tx_col < w):
            raise ValueError(
                f"Tx index {(self.tx_row, self.tx_col)} outside map {(h, w)}"
            )

        tx = np.zeros((h, w), dtype=np.float32)
        tx[self.tx_row, self.tx_col] = 1.0
        e_real, e_imag, e_real_raw, e_imag_raw = compute_einc(
            tx,
            carrier_frequency_hz=self.carrier_frequency_hz,
            dx_m=self.dx_m,
            normalize=True,
        )

        self.tx_mask = tx
        self.einc_norm = np.stack([e_real, e_imag], axis=0).astype(np.float32)
        self.einc_raw = np.stack([e_real_raw, e_imag_raw], axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, idx: int):
        ci = self.cell_ids[idx]
        path = self.cell_dir / f"{ci}.npz"

        with np.load(path) as pack:
            building = pack["building"].astype(np.float32)
            target_norm = pack["target_norm"].astype(np.float32)
            gain_db = pack["gain_db"].astype(np.float32)
            mask = pack["measured_mask"].astype(np.float32)
            rsp_dbm = np.float32(pack["rsp_dbm"])

        x = np.concatenate(
            [
                building[None],
                self.tx_mask[None],
                self.einc_norm,
            ],
            axis=0,
        ).astype(np.float32)

        return {
            "input": torch.from_numpy(x),
            "einc_raw": torch.from_numpy(self.einc_raw.copy()),
            "target_norm": torch.from_numpy(target_norm[None]),
            "gain_db": torch.from_numpy(gain_db[None]),
            "mask": torch.from_numpy(mask[None]),
            "rsp_dbm": torch.tensor(rsp_dbm, dtype=torch.float32),
            "ci": ci,
        }
