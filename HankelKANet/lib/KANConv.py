import math
import torch
import torch.nn as nn
from KANLinear import HankelKANLinear  # 或者改成 from KANLinear import KANLinear as HankelKANLinear
import convolution


class HankelKANConvolution(nn.Module):

    def __init__(
            self,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            orders=(0, 1),
            learnable_k=True,
            z_min=math.pi * 1e-3,
            kappa_min=0.1,
            kappa_max=10.0,
            kappa_init=math.pi,
            eps=1e-6,
            base_activation=nn.SiLU,
            scale_base=1.0,
            scale_hankel=1.0,
            r_min=None,  # 仅为兼容旧接口；新公式不再使用 r_min

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
        self.conv = HankelKANLinear(
            in_features=in_features,
            out_features=1,
            orders=orders,
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




class HankelKANConv2d(nn.Module):


    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            orders=(0, 1),
            learnable_k=True,
            z_min=math.pi * 1e-3,
            kappa_min=0.1,
            kappa_max=10.0,
            kappa_init=math.pi,
            eps=1e-6,
            base_activation=nn.SiLU,
            scale_base=1.0,
            scale_hankel=1.0,
            r_min=None,  # 仅为兼容旧接口；新公式不再使用 r_min
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        convs = []
        # 为每个 (out_c, in_c) 配一个 HankelKANConvolution
        for _out in range(out_channels):
            for _in in range(in_channels):
                convs.append(
                    HankelKANConvolution(
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        orders=orders,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W)
        device = x.device
        return convolution.multiple_convs_kan_conv2d(
            x,
            self.convs,
            kernel_side=self.kernel_size[0],
            out_channels=self.out_channels,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            device=device,
        )