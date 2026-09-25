import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HankelAmplitudeFunction(torch.autograd.Function):


    @staticmethod
    def forward(ctx, z, order: int, eps: float):
        order = int(order)

        # Special functions are evaluated in FP32 for numerical stability.
        z32 = z.float()

        if order == 0:
            J = torch.special.bessel_j0(z32)
            Y = torch.special.bessel_y0(z32)

            # J0'(z) = -J1(z)
            # Y0'(z) = -Y1(z)
            dJ = -torch.special.bessel_j1(z32)
            dY = -torch.special.bessel_y1(z32)

        elif order == 1:
            J = torch.special.bessel_j1(z32)
            Y = torch.special.bessel_y1(z32)

            # J1'(z) = J0(z) - J1(z)/z
            # Y1'(z) = Y0(z) - Y1(z)/z
            dJ = torch.special.bessel_j0(z32) - J / z32
            dY = torch.special.bessel_y0(z32) - Y / z32

        else:
            raise ValueError(
                f"HankelAmplitudeFunction only supports order 0 or 1, got {order}."
            )

        amp = torch.sqrt(J * J + Y * Y)

        # Keep exactly the same feature definition as the current implementation.
        phi = torch.log1p(amp + float(eps))

        # d|H|/dz
        damp_dz = (
            J * dJ + Y * dY
        ) / amp.clamp_min(1e-12)

        # d log(1 + |H| + eps) / dz
        dphi_dz = damp_dz / (
            1.0 + amp + float(eps)
        )

        ctx.save_for_backward(
            dphi_dz.to(dtype=z.dtype)
        )

        return phi.to(dtype=z.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (dphi_dz,) = ctx.saved_tensors

        grad_z = grad_output * dphi_dz

        # z has gradient; order and eps do not.
        return grad_z, None, None

class HankelKANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        orders=(0, 1),           # 主模型固定使用 0、1 阶
        learnable_k: bool = True,
        z_min: float = math.pi * 1e-3,   # Hankel 自变量的固定下界，避免 z=0 奇异
        kappa_min: float = 0.1,           # 归一化径向频率下界
        kappa_max: float = 10.0,          # 归一化径向频率上界
        kappa_init: float = math.pi,      # 与原始代码 k_init=pi 保持一致
        eps: float = 1e-6,                # 幅值里的稳定项
        base_activation=nn.SiLU,
        scale_base: float = 1.0,
        scale_hankel: float = 1.0,
        r_min: float = None,             # 仅为兼容旧 KANConv 接口；新公式不再使用 r_min
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.orders = tuple(orders)
        self.n_orders = len(self.orders)
        if self.orders != (0, 1):
            raise ValueError(
                f"Baseline HankelKANLinear only supports orders=(0, 1), got {self.orders}. "
                f"Use KANLinear_order_ablation.py for Hankel-order ablation experiments."
            )

        self.z_min = float(z_min)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.kappa_init = float(kappa_init)
        self.eps = float(eps)

        self.base_activation = base_activation()
        self.scale_hankel = float(scale_hankel)

        if self.z_min <= 0:
            raise ValueError(f"z_min must be positive, got {self.z_min}.")
        if self.kappa_min <= 0:
            raise ValueError(f"kappa_min must be positive, got {self.kappa_min}.")
        if self.kappa_max <= self.kappa_min:
            raise ValueError(
                f"kappa_max must be greater than kappa_min, "
                f"got [{self.kappa_min}, {self.kappa_max}]."
            )
        if not (self.kappa_min < self.kappa_init < self.kappa_max):
            raise ValueError(
                f"kappa_init must lie strictly inside [kappa_min, kappa_max], "
                f"got kappa_init={self.kappa_init}."
            )

        # 有界归一化径向频率：
        # kappa = kappa_min + (kappa_max-kappa_min) * sigmoid(theta)
        ratio = (
            (self.kappa_init - self.kappa_min)
            / (self.kappa_max - self.kappa_min)
        )
        theta_init = math.log(ratio / (1.0 - ratio))
        theta = torch.tensor(theta_init, dtype=torch.get_default_dtype())

        if learnable_k:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta)

        # Hankel 0阶、1阶两条分支的权重
        self.hankel0_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.hankel1_weight = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):

        # Hankel 0阶
        nn.init.kaiming_uniform_(self.hankel0_weight, a=math.sqrt(5) * self.scale_hankel)
        with torch.no_grad():
            self.hankel0_weight.mul_(self.scale_hankel)

        # Hankel 1阶
        nn.init.kaiming_uniform_(self.hankel1_weight, a=math.sqrt(5) * self.scale_hankel)
        with torch.no_grad():
            self.hankel1_weight.mul_(self.scale_hankel)

    def current_kappa(self) -> torch.Tensor:
        return (
            self.kappa_min
            + (self.kappa_max - self.kappa_min) * torch.sigmoid(self.theta)
        )

    def _hankel_amp(self, order: int, z: torch.Tensor) -> torch.Tensor:
        return HankelAmplitudeFunction.apply(
            z,
            int(order),
            float(self.eps)
        )

    def hankel_bases(self, x: torch.Tensor) -> torch.Tensor:

        assert x.dim() == 2 and x.size(1) == self.in_features

        # x 为无量纲归一化距离 rho(x)=d(x)/L_ref
        rho = x
        kappa = self.current_kappa()

        # Response 定义：z(x)=max{kappa*rho(x), z_min}
        z = (kappa * rho).clamp_min(self.z_min)

        hankel = []
        for o in self.orders:
            phi = self._hankel_amp(o, z)
            hankel.append(phi.unsqueeze(-1))

        return torch.cat(hankel, dim=-1)

    # 前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)          # (B*, F)

        # Hankel 分支
        hankel = self.hankel_bases(x_flat)
        hankel0_out = F.linear(hankel[..., 0], self.hankel0_weight)  # 0阶分支
        hankel1_out = F.linear(hankel[..., 1], self.hankel1_weight)  # 1阶分支
        y = hankel0_out + hankel1_out

        y = y.view(*original_shape[:-1], self.out_features)
        return y