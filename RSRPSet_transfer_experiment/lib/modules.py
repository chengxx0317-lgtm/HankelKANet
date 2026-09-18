import math
import torch
import torch.nn as nn
from KANConv import HankelKANConv2d


# --------- Basic modules ---------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, act=nn.ReLU):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            ConvBNAct(in_ch, mid_ch, k=3, s=1, p=1, act=act),
            ConvBNAct(mid_ch, out_ch, k=3, s=1, p=1, act=act),
        )

    def forward(self, x):
        return self.block(x)


# --------- Physics branch block ---------
class PhysBlock(nn.Module):
    def __init__(
        self,
        in_ch=1,
        phys_ch=32,
        k=3,
        p=1,
        act=nn.SiLU,
        orders=(0, 1),
        learnable_k=True,
        z_min=math.pi * 1e-3,
        kappa_min=0.1,
        kappa_max=10.0,
        kappa_init=math.pi,
    ):
        super().__init__()
        self.hkan = HankelKANConv2d(
            in_channels=in_ch,
            out_channels=phys_ch,
            kernel_size=k,
            stride=1,
            padding=p,
            dilation=1,
            orders=orders,
            learnable_k=learnable_k,
            z_min=z_min,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            kappa_init=kappa_init,
            eps=1e-6,
            base_activation=act,
            scale_base=1.0,
            scale_hankel=1.0,
        )
        self.bn = nn.BatchNorm2d(phys_ch)
        self.act = act(inplace=True)

    def forward(self, x):
        x = self.hkan(x)
        x = self.bn(x)
        return self.act(x)


class HKANNet(nn.Module):
    """
    HankelKANet with size-adaptive decoder.

    The learnable modules and parameter names are intentionally identical to the
    original main-experiment HKANNet.  Only the decoder forward pass is changed:
    every ConvTranspose2d receives ``output_size=skip.size()``.  Therefore an
    existing RadioMapSeer state_dict can be loaded strictly, while input sizes
    such as 40x80 no longer need to be multiples of 16.
    """

    def __init__(
        self,
        base_ch=32,
        phys_ch=32,
        learnable_k=True,
        z_min=math.pi * 1e-3,
        kappa_min=0.1,
        kappa_max=10.0,
        kappa_init=math.pi,
    ):
        super().__init__()

        C = base_ch
        Cp = phys_ch
        act = nn.SiLU
        orders = (0, 1)

        # r_map pooling (kept exactly as in the current main model)
        self.rpool1 = nn.Identity()
        self.rpool2 = nn.AvgPool2d(2)
        self.rpool3 = nn.AvgPool2d(4)
        self.rpool4 = nn.AvgPool2d(8)

        # Main encoder pooling
        self.pool = nn.AvgPool2d(2)

        # ===== Encoder Level 1 =====
        self.stem_img = ConvBNAct(1, C, k=3, s=1, p=1, act=act)
        self.stem_phys = PhysBlock(
            in_ch=1,
            phys_ch=Cp,
            k=3,
            p=1,
            act=act,
            orders=orders,
            learnable_k=learnable_k,
            z_min=z_min,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            kappa_init=kappa_init,
        )
        self.stem_fuse = nn.Conv2d(C + Cp, C, kernel_size=1, bias=False)
        self.enc1 = DoubleConv(C, 2 * C, act=act)

        # ===== Encoder Level 2 =====
        self.phys_l2 = PhysBlock(
            1,
            2 * Cp,
            k=3,
            p=1,
            act=act,
            orders=orders,
            learnable_k=learnable_k,
            z_min=z_min,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            kappa_init=kappa_init,
        )
        self.enc2_fuse = nn.Conv2d(2 * C + 2 * Cp, 2 * C, kernel_size=1, bias=False)
        self.enc2 = DoubleConv(2 * C, 4 * C, act=act)

        # ===== Encoder Level 3 =====
        self.phys_l3 = PhysBlock(
            1,
            4 * Cp,
            k=5,
            p=2,
            act=act,
            orders=orders,
            learnable_k=learnable_k,
            z_min=z_min,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            kappa_init=kappa_init,
        )
        self.enc3_fuse = nn.Conv2d(4 * C + 4 * Cp, 4 * C, kernel_size=1, bias=False)
        self.enc3 = DoubleConv(4 * C, 8 * C, act=act)

        # ===== Encoder Level 4 =====
        self.phys_l4 = PhysBlock(
            1,
            8 * Cp,
            k=5,
            p=2,
            act=act,
            orders=orders,
            learnable_k=learnable_k,
            z_min=z_min,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            kappa_init=kappa_init,
        )
        self.enc4_fuse = nn.Conv2d(8 * C + 8 * Cp, 8 * C, kernel_size=1, bias=False)
        self.enc4 = DoubleConv(8 * C, 16 * C, act=act)

        # ===== Bottleneck =====
        self.bottleneck = DoubleConv(16 * C, 16 * C, act=act)

        # ===== Decoder =====
        # Keep module definitions unchanged so old checkpoints remain strictly compatible.
        self.up4 = nn.ConvTranspose2d(16 * C, 16 * C, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(32 * C, 8 * C, act=act)

        self.up3 = nn.ConvTranspose2d(8 * C, 8 * C, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(16 * C, 4 * C, act=act)

        self.up2 = nn.ConvTranspose2d(4 * C, 4 * C, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(8 * C, 2 * C, act=act)

        self.up1 = nn.ConvTranspose2d(2 * C, 2 * C, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(4 * C, C, act=act)

        self.dec_final = DoubleConv(C, C, act=act)
        self.head = nn.Conv2d(C, 1, kernel_size=1)

    @staticmethod
    def _check_input_size(x: torch.Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(
                f"HKANNet expects input shape (B, 2, H, W), got {tuple(x.shape)}."
            )
        h, w = int(x.shape[-2]), int(x.shape[-1])
        # Four floor-dividing AvgPool2d(2) stages require each dimension >= 16.
        if h < 16 or w < 16:
            raise ValueError(
                f"Input spatial size must be at least 16x16 for four encoder pools; got {h}x{w}."
            )

    def forward(self, x):
        self._check_input_size(x)

        # Split branches
        x_img = x[:, 0:1, :, :]  # building map
        x_r = x[:, 1:2, :, :]    # normalized Tx-Rx radial-distance map

        # ===== Encoder Level 1 =====
        f_img1 = self.stem_img(x_img)
        f_phys1 = self.stem_phys(self.rpool1(x_r))
        f1 = self.stem_fuse(torch.cat([f_img1, f_phys1], dim=1))
        x1 = self.enc1(f1)

        # ===== Encoder Level 2 =====
        x = self.pool(x1)
        r2 = self.rpool2(x_r)
        p2 = self.phys_l2(r2)
        x2 = self.enc2(self.enc2_fuse(torch.cat([x, p2], dim=1)))

        # ===== Encoder Level 3 =====
        x = self.pool(x2)
        r3 = self.rpool3(x_r)
        p3 = self.phys_l3(r3)
        x3 = self.enc3(self.enc3_fuse(torch.cat([x, p3], dim=1)))

        # ===== Encoder Level 4 =====
        x = self.pool(x3)
        r4 = self.rpool4(x_r)
        p4 = self.phys_l4(r4)
        x4 = self.enc4(self.enc4_fuse(torch.cat([x, p4], dim=1)))

        # ===== Bottleneck =====
        x = self.pool(x4)
        xb = self.bottleneck(x)

        # ===== Size-adaptive decoder =====
        # output_size resolves the one-pixel ambiguity introduced by floor pooling
        # for odd / non-2^4-divisible spatial dimensions. It does NOT pad the input.
        x = self.up4(xb, output_size=x4.size())
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x, output_size=x3.size())
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x, output_size=x2.size())
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x, output_size=x1.size())
        x = self.dec1(torch.cat([x, x1], dim=1))

        x = self.dec_final(x)
        return self.head(x)
