import math
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from scipy import special


FREQUENCY = 5.9e9
DX = 1.0
EPSILON0 = 8.8541878128e-12
C0 = 299792458.0


@lru_cache(maxsize=8)
def _discrete_kernel(height, width):
    k = 2 * math.pi * FREQUENCY / C0
    a = DX / math.sqrt(math.pi)
    ka = k * a
    dy = np.arange(2 * height)
    dx = np.arange(2 * width)
    dy = np.where(dy < height, dy, dy - 2 * height)
    dx = np.where(dx < width, dx, dx - 2 * width)
    distance = np.hypot(dy[:, None], dx[None, :]) * DX
    coefficients = np.zeros((2 * height, 2 * width), dtype=np.complex128)
    nonzero = distance > 0
    coefficients[nonzero] = (0.5j * math.pi * k * a * special.j1(ka)) * special.hankel2(0, k * distance[nonzero])
    coefficients[0, 0] = 0.5j * (math.pi * ka * special.hankel2(1, ka) - 2j)
    coefficients[height, :] = 0
    coefficients[:, width] = 0
    if not np.isfinite(coefficients).all():
        raise FloatingPointError('Non-finite VIE coefficients')
    coefficients.setflags(write=False)
    return coefficients


@lru_cache(maxsize=4)
def disk_cell_averaged_incident_lookup(height, width, frequency_hz=FREQUENCY, dx_m=DX):
    if frequency_hz != FREQUENCY or dx_m != DX:
        raise ValueError('Frequency and grid spacing must match the RadioMapSeer experiment')
    k = 2 * math.pi * FREQUENCY / C0
    coefficients = _discrete_kernel(height, width)
    dy = np.arange(-(height - 1), height)
    dx = np.arange(-(width - 1), width)
    average = -coefficients[np.ix_(dy % (2 * height), dx % (2 * width))] / (k * k * DX * DX)
    result = np.asarray(average, dtype=np.complex64)
    result.setflags(write=False)
    return result


class VIEOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectra = {}

    def _spectrum(self, height, width, device):
        key = (height, width, str(device))
        if key not in self.spectra:
            kernel = np.array(_discrete_kernel(height, width))
            tensor = torch.as_tensor(kernel, device=device, dtype=torch.complex64)
            self.spectra[key] = torch.fft.fft2(tensor)
        return self.spectra[key]

    def apply_w(self, source):
        if source.ndim != 3 or not source.is_complex():
            raise ValueError('Expected (B,H,W) complex source')
        batch, height, width = source.shape
        padded = torch.zeros(batch, 2 * height, 2 * width, device=source.device, dtype=source.dtype)
        padded[:, :height, :width] = source
        transformed = torch.fft.fft2(padded) * self._spectrum(height, width, source.device)
        return torch.fft.ifft2(transformed)[:, :height, :width]

    def residual(self, predicted, incident, buildings):
        if predicted.shape != incident.shape or predicted.ndim != 4 or predicted.shape[1] != 2:
            raise ValueError('The predicted and incident fields must have shape (B,2,H,W)')
        if buildings.shape != predicted[:, :1].shape:
            raise ValueError('Invalid building mask shape')
        total = torch.complex(predicted[:, 0].float(), predicted[:, 1].float())
        excitation = torch.complex(incident[:, 0].float(), incident[:, 1].float())
        contrast = 3.0 * buildings[:, 0].float()
        return total + self.apply_w(contrast.to(torch.complex64) * total) - excitation

    @staticmethod
    def physical_loss(residual):
        if not torch.isfinite(residual).all():
            raise FloatingPointError('Non-finite physical residual')
        count = residual.shape[-1] * residual.shape[-2]
        return torch.linalg.vector_norm(residual.flatten(1), dim=1).div(count).mean()
