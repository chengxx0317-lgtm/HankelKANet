# KANConv_angular.py

import math
import torch.nn.functional as F
import torch
import torch.nn as nn

from KANLinear_angular import AngularHankelKANLinear


def calc_out_dims(
    matrix: torch.Tensor,
    kernel_side: int,
    stride,
    dilation,
    padding,
):
    _, _, H, W = matrix.shape

    h_out = (
        math.floor(
            (
                H
                + 2 * padding[0]
                - kernel_side
                - (kernel_side - 1) * (dilation[0] - 1)
            )
            / stride[0]
        )
        + 1
    )

    w_out = (
        math.floor(
            (
                W
                + 2 * padding[1]
                - kernel_side
                - (kernel_side - 1) * (dilation[1] - 1)
            )
            / stride[1]
        )
        + 1
    )

    return h_out, w_out


class AngularHankelKANConvolution(nn.Module):
    """One KxK angular Hankel-KAN kernel for one input/output channel pair."""

    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        max_order=1,
        learnable_k=True,
        z_min=math.pi * 1e-3,
        kappa_min=0.1,
        kappa_max=10.0,
        kappa_init=math.pi,
        eps=1e-6,
        base_activation=nn.SiLU,
        scale_base=1.0,
        scale_hankel=1.0,
        r_min=None,  # compatibility only; not used
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        in_features = math.prod(kernel_size)

        self.conv = AngularHankelKANLinear(
            in_features=in_features,
            out_features=1,
            max_order=max_order,
            learnable_k=learnable_k,
            z_min=z_min,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            kappa_init=kappa_init,
            eps=eps,
            base_activation=base_activation,
            scale_base=scale_base,
            scale_hankel=scale_hankel,
            r_min=r_min,
        )


class AngularHankelKANConv2d(nn.Module):
    """
    Angular counterpart of HankelKANConv2d.

    rho and phi are unfolded using exactly the same spatial kernel so every
    radial sample remains paired with its Tx-centered azimuth sample.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        max_order=1,
        learnable_k=True,
        z_min=math.pi * 1e-3,
        kappa_min=0.1,
        kappa_max=10.0,
        kappa_init=math.pi,
        eps=1e-6,
        base_activation=nn.SiLU,
        scale_base=1.0,
        scale_hankel=1.0,
        r_min=None,  # compatibility only; not used
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.max_order = int(max_order)

        convs = []
        for _out in range(self.out_channels):
            for _in in range(self.in_channels):
                convs.append(
                    AngularHankelKANConvolution(
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        max_order=max_order,
                        learnable_k=learnable_k,
                        z_min=z_min,
                        kappa_min=kappa_min,
                        kappa_max=kappa_max,
                        kappa_init=kappa_init,
                        eps=eps,
                        base_activation=base_activation,
                        scale_base=scale_base,
                        scale_hankel=scale_hankel,
                        r_min=r_min,
                    )
                )

        self.convs = nn.ModuleList(convs)

    def forward(
        self,
        rho: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        if rho.shape != phi.shape:
            raise ValueError(
                f"rho and phi must have identical shapes, got "
                f"{tuple(rho.shape)} and {tuple(phi.shape)}"
            )

        B, C_in, _, _ = rho.shape

        if C_in != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {C_in}"
            )

        h_out, w_out = calc_out_dims(
            rho,
            kernel_side=self.kernel_size[0],
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
        )

        # ------------------------------------------------------------
        # Boundary padding for angular Hankel features.
        #
        # Do not use zero padding:
        # rho=0 would be interpreted as a near-source location,
        # while phi=0 would introduce an artificial angular direction.
        # ------------------------------------------------------------

        pad_h, pad_w = self.padding

        if pad_h > 0 or pad_w > 0:

            rho_for_unfold = F.pad(
                rho,
                pad=(pad_w, pad_w, pad_h, pad_h),
                mode="replicate",
            )

            phi_for_unfold = F.pad(
                phi,
                pad=(pad_w, pad_w, pad_h, pad_h),
                mode="replicate",
            )

        else:
            rho_for_unfold = rho
            phi_for_unfold = phi

        # Padding has already been performed manually.
        unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=(0, 0),
            stride=self.stride,
        )

        rho_unfold = unfold(rho_for_unfold)
        phi_unfold = unfold(phi_for_unfold)
        L = rho_unfold.shape[-1]
        K2 = self.kernel_size[0] * self.kernel_size[1]

        rho_unfold = rho_unfold.view(B, C_in, K2, L).permute(0, 1, 3, 2)
        phi_unfold = phi_unfold.view(B, C_in, K2, L).permute(0, 1, 3, 2)

        matrix_out = torch.zeros(
            (B, self.out_channels, h_out, w_out),
            device=rho.device,
            dtype=rho.dtype,
        )

        for out_c in range(self.out_channels):
            out_accum = torch.zeros(
                (B, L),
                device=rho.device,
                dtype=rho.dtype,
            )

            for in_c in range(C_in):
                k_idx = out_c * C_in + in_c
                kernel = self.convs[k_idx]

                rho_patches = rho_unfold[:, in_c, :, :].reshape(B * L, K2)
                phi_patches = phi_unfold[:, in_c, :, :].reshape(B * L, K2)

                conv_result = kernel.conv(rho_patches, phi_patches)
                conv_result = conv_result.view(B, L)

                out_accum += conv_result

            matrix_out[:, out_c, :, :] = out_accum.view(B, h_out, w_out)

        return matrix_out