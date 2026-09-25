import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import math
import torch.nn.functional as F


# 基础卷积模块

def conv3x3(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=s, padding=p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c)
        self.conv2 = conv3x3(out_c, out_c)
    def forward(self, x):
        return self.conv2(self.conv1(x))



# U-Net (Physics-driven)

class UNet_Physics(nn.Module):
 #输入4通道 ,输出2通道复电场
    def __init__(self, in_c=4, out_c=2):
        super().__init__()
        # 编码
        self.e1 = DoubleConv(in_c, 128)
        self.e2 = DoubleConv(128, 256)
        self.e3 = DoubleConv(256, 512)
        self.e4 = DoubleConv(512, 1024)

        # 下采样
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(1024, 1024)
        # 解码
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


    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        xb = self.bottleneck(self.pool(e4))
        d4 = self.d4(torch.cat([self.up4(xb), e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        head = self.head(d1)
        out =  self.out(head)
        return out  # (B,2,H,W)



# 物理算子

class PhysicsOperator(nn.Module):

    def __init__(self, k, dx):
        super().__init__()
        self.k = k
        self.dx = dx

    def build_kernel(self, size):
        h, w = size
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        cy, cx = h // 2, w // 2
        r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * self.dx + 1e-6

        kernel = torch.exp(-1j * self.k * r) / torch.sqrt(r)

        kernel = kernel / torch.sum(torch.abs(kernel))
        return kernel.to(torch.complex64)

    def forward(self, E_pred, E_inc_raw, chi):

        B, _, H, W = E_pred.shape
        E_complex = E_pred[:,0,:,:] + 1j * E_pred[:,1,:,:]
        K = self.build_kernel((H, W)).to(E_complex.device)

        K_fft = fft.fft2(K)
        # 计算 chi*E
        chiE = chi.squeeze(1) * E_complex
        pad = (W // 2, W // 2, H // 2, H // 2)  # 左、右、上、下各填充半尺寸
        chiE_pad = F.pad(chiE, pad)

        conv_term = fft.ifft2(fft.fft2(chiE_pad) * fft.fft2(K, s=chiE_pad.shape[-2:]))

        conv_term = conv_term[..., pad[2]:-pad[3], pad[0]:-pad[1]]

        phys_term = E_complex + conv_term - (E_inc_raw[:, 0] + 1j * E_inc_raw[:, 1])
        loss_phy = torch.mean(torch.abs(phys_term) ** 2)
        return loss_phy



# 复电场 → 初始路径损耗
def field_to_PL(E_pred, pl_min_db=-11.0, pl_max_db=70.0, eps=1e-9):

    # 幅值  dB
    E_amp = torch.sqrt(E_pred[:,0]**2 + E_pred[:,1]**2 + eps)
    PL_db = -20.0 * torch.log10(E_amp + eps)

    # 全局归一化
    PL_db = torch.clamp(PL_db, pl_min_db, pl_max_db)
    PL = (PL_db - pl_min_db) / (pl_max_db - pl_min_db)
    PL = 1 - PL
    return PL.unsqueeze(1)




# U-Net (Data-driven 数据驱动)

class UNet_Data(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        # 编码
        self.e1 = DoubleConv(in_c, 128)
        self.e2 = DoubleConv(128, 256)
        self.e3 = DoubleConv(256, 512)
        self.e4 = DoubleConv(512, 1024)

        # 下采样
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(1024, 1024)
        # 解码
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

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        xb = self.bottleneck(self.pool(e4))
        d4 = self.d4(torch.cat([self.up4(xb), e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        head = self.head(d1)
        out =  self.out(head)
        return out  # (B,2,H,W)



# 整体模型

class PhysicsInformedModel(nn.Module):
    def __init__(self, k, dx, lambda_d=1.0):
        super().__init__()
        self.unet_phy = UNet_Physics()
        self.unet_data = UNet_Data()
        self.physics = PhysicsOperator(k, dx)
        self.lambda_d = lambda_d

    def forward(self, inputs, E_inc_raw, chi, label=None):
        # 第一阶段：预测复电场
        E_pred = self.unet_phy(inputs)
        # 物理约束损失（使用未归一化的 E_inc_raw）
        loss_phy = self.physics(E_pred, E_inc_raw, chi)
        # 转换为初始路径损耗
        PL_init = field_to_PL(E_pred)
        bld = inputs[:, 0:1, :, :]
        tx = inputs[:, 1:2, :, :]
        pl_input = torch.cat([bld, tx, PL_init], dim=1)
        # 第二阶段：预测最终路径损耗
        PL_pred = self.unet_data(pl_input)

        # 数据损失
        loss_data = 0.0
        if label is not None:
            loss_data = torch.mean((PL_pred - label)**2)
        loss_total = self.lambda_d * loss_data + loss_phy
        return {
            "E_pred": E_pred,
            "PL_init": PL_init,
            "PL_pred": PL_pred,
            "loss_phy": loss_phy,
            "loss_data": loss_data,
            "loss_total": loss_total
        }