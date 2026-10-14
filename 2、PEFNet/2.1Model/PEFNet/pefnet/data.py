from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .vie import disk_cell_averaged_incident_lookup


def scenario_partitions():
    indices = np.arange(700)
    np.random.RandomState(42).shuffle(indices)
    return {'train': indices[:500].tolist(), 'val': indices[500:600].tolist(), 'test': indices[600:].tolist()}


def read_image(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('L'), dtype=np.float32)
    if array.max() > 1:
        array /= 255.0
    return array


class RadioMapSeer(Dataset):
    def __init__(self, root, split):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Invalid split')
        self.root = Path(root)
        self.ids = scenario_partitions()[split]
        self.building_dir = self.root / 'png' / 'buildings_complete'
        self.tx_dir = self.root / 'png' / 'antennas'
        self.label_dir = self.root / 'gain' / 'DPM'
        for directory in (self.building_dir, self.tx_dir, self.label_dir):
            if not directory.is_dir():
                raise FileNotFoundError(directory)

    def __len__(self):
        return len(self.ids) * 80

    def __getitem__(self, index):
        scene = self.ids[index // 80]
        ident = f'{scene}_{index % 80}'
        building = read_image(self.building_dir / f'{scene}.png')
        transmitter = read_image(self.tx_dir / f'{ident}.png')
        target = read_image(self.label_dir / f'{ident}.png')
        if building.shape != transmitter.shape or building.shape != target.shape:
            raise ValueError(f'Inconsistent grid dimensions: {ident}')
        if building.shape != (256, 256):
            raise ValueError(f'Expected 256x256 map: {ident}')
        locations = np.argwhere(transmitter > 0.5)
        if len(locations) == 0:
            raise ValueError(f'Missing transmitter: {ident}')
        ty, tx = locations[0]
        h, w = building.shape
        lookup = disk_cell_averaged_incident_lookup(h, w, frequency_hz=5.9e9, dx_m=1.0)
        incident = lookup[np.ix_(np.arange(h) - ty + h - 1, np.arange(w) - tx + w - 1)]
        real = incident.real
        imag = incident.imag
        real_normalized = (real - real.min()) / (real.max() - real.min() + 1e-9)
        imag_normalized = (imag - imag.min()) / (imag.max() - imag.min() + 1e-9)
        inputs = np.stack((transmitter, building, real_normalized, imag_normalized)).astype(np.float32)
        raw = np.stack((real, imag)).astype(np.float32)
        return {
            'inputs': torch.from_numpy(inputs),
            'incident': torch.from_numpy(raw),
            'target': torch.from_numpy(target[None].copy()),
            'outdoor': torch.from_numpy((building[None] < 0.5).copy()),
            'id': ident,
        }
