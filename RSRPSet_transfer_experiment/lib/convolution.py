import torch
import numpy as np
from typing import Tuple, List
import torch.nn.functional as F

def calc_out_dims(
    matrix: torch.Tensor,
    kernel_side: int,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    padding: Tuple[int, int],
):

    batch_size, n_channels, H, W = matrix.shape

    h_out = (
        np.floor(
            (
                H
                + 2 * padding[0]
                - kernel_side
                - (kernel_side - 1) * (dilation[0] - 1)
            )
            / stride[0]
        ).astype(int)
        + 1
    )
    w_out = (
        np.floor(
            (
                W
                + 2 * padding[1]
                - kernel_side
                - (kernel_side - 1) * (dilation[1] - 1)
            )
            / stride[1]
        ).astype(int)
        + 1
    )

    return h_out, w_out, batch_size, n_channels


def multiple_convs_kan_conv2d(
    matrix: torch.Tensor,              # (B, C_in, H, W)
    kernels: List,                     # len = C_out * C_in，每个有 .conv: HankelKANLinear
    kernel_side: int,
    out_channels: int,
    stride: Tuple[int, int] = (1, 1),
    dilation: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    device = "cuda",
) -> torch.Tensor:

    B, C_in, H, W = matrix.shape
    h_out, w_out, _, _ = calc_out_dims(
        matrix, kernel_side, stride, dilation, padding
    )

    matrix_out = torch.zeros(
        (B, out_channels, h_out, w_out), device=device, dtype=matrix.dtype
    )

    # 使用 unfold 提取所有 patch
    # Hankel branch boundary padding


    pad_h, pad_w = padding

    if pad_h > 0 or pad_w > 0:
        matrix_for_unfold = F.pad(
            matrix,
            pad=(pad_w, pad_w, pad_h, pad_h),
            mode="replicate",
        )
    else:
        matrix_for_unfold = matrix

    # 使用 unfold 提取所有 patch
    # 注意：padding 已经在上面手动完成，因此这里必须设为 0
    unfold = torch.nn.Unfold(
        kernel_size=(kernel_side, kernel_side),
        dilation=dilation,
        padding=(0, 0),
        stride=stride,
    )

    # unfolded: (B, C_in * K*K, L)
    unfolded = unfold(matrix_for_unfold)
    L = unfolded.shape[-1]
    # 重排成 (B, C_in, L, K*K)
    unfolded = unfolded.view(B, C_in, kernel_side * kernel_side, L).permute(
        0, 1, 3, 2
    )  # (B, C_in, L, K*K)

    # 遍历输出通道
    for out_c in range(out_channels):
        out_accum = torch.zeros((B, L), device=device, dtype=matrix.dtype)

        for in_c in range(C_in):
            k_idx = out_c * C_in + in_c
            kernel = kernels[k_idx]  # HankelKANConvolution
            # 取对应输入通道的所有 patch: (B, L, K*K)
            patches = unfolded[:, in_c, :, :]             # (B, L, K*K)
            patches = patches.reshape(B * L, -1)          # (B*L, K*K)

            # 送入 kernel.conv（HankelKANLinear）
            conv_result = kernel.conv(patches)            # (B*L, 1)
            conv_result = conv_result.view(B, L)          # (B, L)

            out_accum += conv_result

        # reshape 成 (B, h_out, w_out)
        matrix_out[:, out_c, :, :] = out_accum.view(B, h_out, w_out)

    return matrix_out
