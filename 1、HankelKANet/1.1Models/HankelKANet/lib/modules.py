import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            ConvBNAct(mid_ch, out_ch, k=5, s=1, p=2, act=act),
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

        # r_map pooling (kept as in original HKANNet)
        self.rpool1 = nn.Identity()
        self.rpool2 = nn.AvgPool2d(2)
        self.rpool3 = nn.AvgPool2d(4)
        self.rpool4 = nn.AvgPool2d(8)

        # Main encoder pooling: AvgPool2d(2) (your requested change)
        self.pool = nn.AvgPool2d(2)

        # ===== Encoder Level 1 (H, W) =====
        self.stem_img = ConvBNAct(1, C, k=3, s=1, p=1, act=act)
        self.stem_phys = PhysBlock(
            in_ch=1, phys_ch=Cp, k=3, p=1, act=act, orders=orders,
            learnable_k=learnable_k, z_min=z_min,
            kappa_min=kappa_min, kappa_max=kappa_max, kappa_init=kappa_init
        )
        self.stem_fuse = nn.Conv2d(C + Cp, C, kernel_size=1, bias=False)
        self.enc1 = DoubleConv(C, 2 * C, act=act)  # skip1: 2C

        # Level 2 (2C)
        self.phys_l2 = PhysBlock(
            1, 2 * Cp, k=3, p=1,  act=act, orders=orders,
            learnable_k=learnable_k, z_min=z_min,
            kappa_min=kappa_min, kappa_max=kappa_max, kappa_init=kappa_init
        )
        self.enc2_fuse = nn.Conv2d(2 * C + 2 * Cp, 2 * C, kernel_size=1, bias=False)
        self.enc2 = DoubleConv(2 * C, 4 * C,  act=act)  # skip2: 4C

        # ===== Encoder Level 3 (H/4, W/4) =====
        self.phys_l3 = PhysBlock(
            1, 4 * Cp, k=5, p=2, act=act, orders=orders,
            learnable_k=learnable_k, z_min=z_min,
            kappa_min=kappa_min, kappa_max=kappa_max, kappa_init=kappa_init
        )
        self.enc3_fuse = nn.Conv2d(4 * C + 4 * Cp, 4 * C, kernel_size=1, bias=False)
        self.enc3 = DoubleConv(4 * C, 8 * C,  act=act)  # skip3: 8C

        # ===== Encoder Level 4 (H/8, W/8) =====
        self.phys_l4 = PhysBlock(
            1, 8 * Cp, k=5, p=2, act=act, orders=orders,
            learnable_k=learnable_k, z_min=z_min,
            kappa_min=kappa_min, kappa_max=kappa_max, kappa_init=kappa_init
        )
        self.enc4_fuse = nn.Conv2d(8 * C + 8 * Cp, 8 * C, kernel_size=1, bias=False)
        self.enc4 = DoubleConv(8 * C, 16 * C, act=act)  # skip4: 16C

        # ===== Bottleneck (H/16, W/16) =====
        self.bottleneck = DoubleConv(16 * C, 16 * C, act=act)

        # ===== Decoder (ConvTranspose2d style, consistent with UNet_Physics) =====
        self.up4 = nn.ConvTranspose2d(16 * C, 16 * C, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(32 * C, 8 * C, act=act)

        self.up3 = nn.ConvTranspose2d(8 * C, 8 * C, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(16 * C, 4 * C, act=act)

        self.up2 = nn.ConvTranspose2d(4 * C, 4 * C, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(8 * C, 2 * C,act=act)

        self.up1 = nn.ConvTranspose2d(2 * C, 2*C, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(4 * C, 1*C, act=act)

        # Optional final refinement (kept from original HKANNet)
        self.dec_final = DoubleConv(C, C, act=act)
        self.head = nn.Conv2d(C, 1, kernel_size=1)



    def forward(self, x):
        # Split branches
        x_img = x[:, 0:1, :, :]  # bld
        x_r = x[:, 1:2, :, :]  # r_map

        # ===== Encoder Level 1 =====
        f_img1 = self.stem_img(x_img)               # (B,C,H,W)
        f_phys1 = self.stem_phys(self.rpool1(x_r))  # (B,Cp,H,W)
        f1 = self.stem_fuse(torch.cat([f_img1, f_phys1], dim=1))
        x1 = self.enc1(f1)                          # skip1: (B,C,H,W)

        # ===== Encoder Level 2 =====
        x = self.pool(x1)                           # (B,C,H/2,W/2)
        r2 = self.rpool2(x_r)                       # (B,1,H/2,W/2)
        p2 = self.phys_l2(r2)                       # (B,Cp,H/2,W/2)
        x2 = self.enc2(self.enc2_fuse(torch.cat([x, p2], dim=1)))  # skip2: (B,2C,H/2,W/2)

        # ===== Encoder Level 3 =====
        x = self.pool(x2)                           # (B,2C,H/4,W/4)
        r3 = self.rpool3(x_r)                       # (B,1,H/4,W/4)
        p3 = self.phys_l3(r3)                       # (B,Cp,H/4,W/4)
        x3 = self.enc3(self.enc3_fuse(torch.cat([x, p3], dim=1)))  # skip3: (B,4C,H/4,W/4)

        # ===== Encoder Level 4 =====
        x = self.pool(x3)                           # (B,4C,H/8,W/8)
        r4 = self.rpool4(x_r)                       # (B,1,H/8,W/8)
        p4 = self.phys_l4(r4)                       # (B,Cp,H/8,W/8)
        x4 = self.enc4(self.enc4_fuse(torch.cat([x, p4], dim=1)))  # skip4: (B,8C,H/8,W/8)

        # ===== Bottleneck =====
        x = self.pool(x4)                           # (B,8C,H/16,W/16)
        xb = self.bottleneck(x)                     # (B,16C,H/16,W/16)

        # ===== Decoder =====
        x = self.up4(xb)                            # (B,8C,H/8,W/8)
        x = self.dec4(torch.cat([x, x4], dim=1))    # (B,8C,H/8,W/8)

        x = self.up3(x)                            # (B,4C,H/4,W/4)
        x = self.dec3(torch.cat([x, x3], dim=1))    # (B,4C,H/4,W/4)

        x = self.up2(x)                             # (B,2C,H/2,W/2)
        x = self.dec2(torch.cat([x, x2], dim=1))    # (B,2C,H/2,W/2)

        x = self.up1(x)                             # (B,C,H,W)
        x = self.dec1(torch.cat([x, x1], dim=1))    # (B,C,H,W)

        x = self.dec_final(x)                       # (B,C,H,W)
        out = self.head(x)                          # (B,1,H,W)
        return out