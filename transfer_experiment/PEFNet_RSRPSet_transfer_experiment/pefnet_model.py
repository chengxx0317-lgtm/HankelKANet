from __future__ import annotations

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F


def conv3x3(in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=s, padding=p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c)
        self.conv2 = conv3x3(out_c, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class UNet_Physics(nn.Module):
    def __init__(self, in_c: int = 4, out_c: int = 2):
        super().__init__()
        self.e1 = DoubleConv(in_c, 128)
        self.e2 = DoubleConv(128, 256)
        self.e3 = DoubleConv(256, 512)
        self.e4 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(1024, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 1024, 2, stride=2)
        self.d4 = DoubleConv(2048, 512)
        self.up3 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.d3 = DoubleConv(1024, 256)
        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.d2 = DoubleConv(512, 128)
        self.up1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.d1 = DoubleConv(256, 128)
        self.head = nn.Conv2d(128, 64, 1)
        self.out = nn.Conv2d(64, out_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        xb = self.bottleneck(self.pool(e4))

        d4 = self.d4(torch.cat([self.up4(xb, output_size=e4.size()), e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4, output_size=e3.size()), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3, output_size=e2.size()), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2, output_size=e1.size()), e1], dim=1))
        return self.out(self.head(d1))


class PhysicsOperator(nn.Module):
    def __init__(self, k: float, dx: float):
        super().__init__()
        self.k = float(k)
        self.dx = float(dx)

    def build_kernel(self, size: tuple[int, int], device: torch.device) -> torch.Tensor:
        h, w = size
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        cy, cx = h // 2, w // 2
        r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).float() * self.dx + 1e-6
        kernel = torch.exp(-1j * self.k * r) / torch.sqrt(r)
        kernel = kernel / torch.sum(torch.abs(kernel)).clamp_min(1e-12)
        return kernel.to(torch.complex64)

    def forward(
        self,
        e_pred: torch.Tensor,
        e_inc_raw: torch.Tensor,
        chi: torch.Tensor,
    ) -> torch.Tensor:
        _, _, h, w = e_pred.shape
        e_complex = e_pred[:, 0].float() + 1j * e_pred[:, 1].float()
        kernel = self.build_kernel((h, w), e_complex.device)

        chi_e = chi.squeeze(1).float() * e_complex
        pad = (w // 2, w // 2, h // 2, h // 2)
        chi_e_pad = F.pad(chi_e, pad)

        conv_term = fft.ifft2(
            fft.fft2(chi_e_pad) *
            fft.fft2(kernel, s=chi_e_pad.shape[-2:])
        )
        conv_term = conv_term[..., pad[2]:-pad[3], pad[0]:-pad[1]]

        e_inc_complex = e_inc_raw[:, 0].float() + 1j * e_inc_raw[:, 1].float()
        phys_term = e_complex + conv_term - e_inc_complex
        return torch.mean(torch.abs(phys_term) ** 2)


def field_to_pl(
    e_pred: torch.Tensor,
    pl_min_db: float = -11.0,
    pl_max_db: float = 70.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    e_amp = torch.sqrt(e_pred[:, 0].float() ** 2 + e_pred[:, 1].float() ** 2 + eps)
    pl_db = -20.0 * torch.log10(e_amp + eps)
    pl_db = torch.clamp(pl_db, pl_min_db, pl_max_db)
    pl = (pl_db - pl_min_db) / (pl_max_db - pl_min_db)
    return (1.0 - pl).unsqueeze(1)


class UNet_Data(nn.Module):
    def __init__(self, in_c: int = 3, out_c: int = 1):
        super().__init__()
        self.e1 = DoubleConv(in_c, 128)
        self.e2 = DoubleConv(128, 256)
        self.e3 = DoubleConv(256, 512)
        self.e4 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(1024, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 1024, 2, stride=2)
        self.d4 = DoubleConv(2048, 512)
        self.up3 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.d3 = DoubleConv(1024, 256)
        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.d2 = DoubleConv(512, 128)
        self.up1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.d1 = DoubleConv(256, 128)
        self.head = nn.Conv2d(128, 64, 1)
        self.out = nn.Conv2d(64, out_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        xb = self.bottleneck(self.pool(e4))

        d4 = self.d4(torch.cat([self.up4(xb, output_size=e4.size()), e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4, output_size=e3.size()), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3, output_size=e2.size()), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2, output_size=e1.size()), e1], dim=1))
        return self.out(self.head(d1))


class PhysicsInformedModel(nn.Module):
    def __init__(
        self,
        k: float,
        dx: float,
        lambda_d: float = 1.0,
        lambda_phy: float =1.0,
    ):
        super().__init__()
        self.unet_phy = UNet_Physics()
        self.unet_data = UNet_Data()
        self.physics = PhysicsOperator(k, dx)
        self.lambda_d = float(lambda_d)
        self.lambda_phy = float(lambda_phy)

    def forward(
        self,
        inputs: torch.Tensor,
        e_inc_raw: torch.Tensor,
        chi: torch.Tensor,
        label: torch.Tensor | None = None,
        compute_physics_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        e_pred = self.unet_phy(inputs)
        pl_init = field_to_pl(e_pred)

        building = inputs[:, 0:1]
        tx = inputs[:, 1:2]
        pl_input = torch.cat([building, tx, pl_init], dim=1)
        pl_pred = self.unet_data(pl_input)

        if compute_physics_loss:
            loss_phy = self.physics(e_pred, e_inc_raw, chi)
        else:
            loss_phy = pl_pred.new_zeros(())

        if label is None:
            loss_data = pl_pred.new_zeros(())
        else:
            loss_data = torch.mean((pl_pred.float() - label.float()) ** 2)

        loss_total = self.lambda_d * loss_data + self.lambda_phy * loss_phy

        return {
            "E_pred": e_pred,
            "PL_init": pl_init,
            "PL_pred": pl_pred,
            "loss_phy": loss_phy,
            "loss_data": loss_data,
            "loss_total": loss_total,
        }
