from __future__ import annotations

import os
import warnings

import numpy as np
from scipy import special
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms


warnings.filterwarnings("ignore")


def compute_einc(
    tx_mask: np.ndarray,
    f: float = 5.9e9,
    dx: float = 1.0,
    normalize: bool = True,
):
    c = 3e8
    k = 2.0 * np.pi * float(f) / c
    h, w = tx_mask.shape

    y0, x0 = np.where(tx_mask > 0)
    if len(x0) == 0:
        raise ValueError("Tx mask contains no transmitter pixel.")
    y0, x0 = int(y0[0]), int(x0[0])

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2) * float(dx) + 1e-8

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


class DatasetRadioMapSeer(Dataset):
    def __init__(
        self,
        phase: str,
        dir_dataset: str,
        num_tx: int = 80,
        img_size: int = 256,
        f: float = 5.9e9,
        dx: float = 1.0,
        tensor_transform=None,
    ):
        self.phase = phase
        self.dir_dataset = dir_dataset
        self.dir_buildings = os.path.join(dir_dataset, "png/buildings_complete")
        self.dir_tx = os.path.join(dir_dataset, "png/antennas")
        self.dir_gain = os.path.join(dir_dataset, "gain/DPM")
        self.num_tx = int(num_tx)
        self.f = float(f)
        self.dx = float(dx)
        self.img_size = int(img_size)
        self.tensor_transform = tensor_transform or transforms.ToTensor()

        all_inds = np.arange(700)
        rng = np.random.RandomState(42)
        rng.shuffle(all_inds)

        if phase == "train":
            self.map_inds = all_inds[:500]
        elif phase == "val":
            self.map_inds = all_inds[500:600]
        elif phase == "test":
            self.map_inds = all_inds[600:]
        else:
            raise ValueError("phase must be 'train', 'val', or 'test'.")

    def __len__(self) -> int:
        return len(self.map_inds) * self.num_tx

    def __getitem__(self, idx: int):
        idx_r = idx // self.num_tx
        idx_c = idx % self.num_tx
        map_ind = int(self.map_inds[idx_r])

        building_name = f"{map_ind}.png"
        sample_name = f"{map_ind}_{idx_c}.png"

        building = io.imread(os.path.join(self.dir_buildings, building_name))
        tx = io.imread(os.path.join(self.dir_tx, sample_name))
        gain = np.expand_dims(io.imread(os.path.join(self.dir_gain, sample_name)), axis=2)

        building = np.asarray(building, dtype=np.float32)
        tx = np.asarray(tx, dtype=np.float32)
        gain = np.asarray(gain, dtype=np.float32)

        if building.max() > 1.0:
            building /= 255.0
        if tx.max() > 1.0:
            tx /= 255.0
        if gain.max() > 1.0:
            gain /= 255.0

        if self.img_size != building.shape[0]:
            building = transform.resize(
                building,
                (self.img_size, self.img_size),
                order=0,
                preserve_range=True,
            )
            tx = transform.resize(
                tx,
                (self.img_size, self.img_size),
                order=0,
                preserve_range=True,
            )
            gain = transform.resize(
                gain,
                (self.img_size, self.img_size),
                order=1,
                preserve_range=True,
            )

        e_real, e_imag, e_real_raw, e_imag_raw = compute_einc(
            tx,
            f=self.f,
            dx=self.dx,
            normalize=True,
        )

        inputs = np.stack([building, tx, e_real, e_imag], axis=2)
        e_inc_raw = np.stack([e_real_raw, e_imag_raw], axis=2)

        inputs = self.tensor_transform(inputs).float()
        e_inc_raw = self.tensor_transform(e_inc_raw).float()
        gain = self.tensor_transform(gain).float()

        sample_id = f"{map_ind}_{idx_c}"
        return inputs, gain, e_inc_raw, sample_id
