import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableBesselJY(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, order: int):
        order = int(order)

        z32 = z.float()
        z_safe = z32.clamp_min(1e-12)

        # ----- J0, J1, Y0, Y1 -----
        j0 = torch.special.bessel_j0(z_safe)
        j1 = torch.special.bessel_j1(z_safe)
        y0 = torch.special.bessel_y0(z_safe)
        y1 = torch.special.bessel_y1(z_safe)

        if order == 0:
            J = j0
            Y = y0

            # J0' = -J1
            # Y0' = -Y1
            dJ = -j1
            dY = -y1

        elif order == 1:
            J = j1
            Y = y1

            # J1' = J0 - J1/z
            # Y1' = Y0 - Y1/z
            dJ = j0 - j1 / z_safe
            dY = y0 - y1 / z_safe

        else:
            # Build J_n, Y_n using recurrence
            j_prev = j0
            j_curr = j1

            y_prev = y0
            y_curr = y1

            for n in range(1, order):
                coef = (2.0 * n) / z_safe

                j_next = coef * j_curr - j_prev
                y_next = coef * y_curr - y_prev

                j_prev, j_curr = j_curr, j_next
                y_prev, y_curr = y_curr, y_next

            # At this point:
            # j_curr = J_order
            # j_prev = J_(order-1)
            J = j_curr
            Y = y_curr

            # J_n' = J_(n-1) - n/z * J_n
            # Y_n' = Y_(n-1) - n/z * Y_n
            dJ = j_prev - order * J / z_safe
            dY = y_prev - order * Y / z_safe

        ctx.save_for_backward(
            dJ.to(dtype=z.dtype),
            dY.to(dtype=z.dtype),
        )

        return (
            J.to(dtype=z.dtype),
            Y.to(dtype=z.dtype),
        )

    @staticmethod
    def backward(ctx, grad_J, grad_Y):
        dJ, dY = ctx.saved_tensors

        grad_z = grad_J * dJ + grad_Y * dY

        return grad_z, None
class HankelKANLinear(nn.Module):


    def __init__(
        self,
        in_features: int,
        out_features: int,
        orders=(0, 1),
        learnable_k: bool = True,
        z_min: float = math.pi * 1e-3,
        kappa_min: float = 0.1,
        kappa_max: float = 10.0,
        kappa_init: float = math.pi,
        eps: float = 1e-6,
        base_activation=nn.SiLU,
        scale_base: float = 1.0,
        scale_hankel: float = 1.0,
        r_min: float = None,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        orders = tuple(int(o) for o in orders)
        if len(orders) == 0:
            raise ValueError("orders must contain at least one Hankel order.")
        if any(o < 0 for o in orders):
            raise ValueError(f"Only non-negative integer orders are supported, got {orders}.")
        if len(set(orders)) != len(orders):
            raise ValueError(f"Duplicate Hankel orders are not allowed, got {orders}.")

        self.orders = orders
        self.n_orders = len(self.orders)

        self.z_min = float(z_min)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.kappa_init = float(kappa_init)
        self.eps = float(eps)

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

        # 与主模型完全一致的有界归一化径向频率参数化
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

        # One learnable weight matrix for every selected Hankel order.
        self.hankel_weights = nn.ParameterDict({
            str(order): nn.Parameter(torch.empty(out_features, in_features))
            for order in self.orders
        })

        self.reset_parameters()

    def reset_parameters(self):
        for order in self.orders:
            weight = self.hankel_weights[str(order)]
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5) * self.scale_hankel)
            with torch.no_grad():
                weight.mul_(self.scale_hankel)

    def current_kappa(self) -> torch.Tensor:
        return (
            self.kappa_min
            + (self.kappa_max - self.kappa_min) * torch.sigmoid(self.theta)
        )

    @staticmethod
    def _bessel_jy_integer_order(order: int, z: torch.Tensor):
        return DifferentiableBesselJY.apply(
            z,
            int(order),
        )

    def _hankel_amp(self, order: int, z: torch.Tensor) -> torch.Tensor:
        J, Y = self._bessel_jy_integer_order(int(order), z)
        amp = torch.sqrt(J * J + Y * Y)  # |H_order(z)|
        return torch.log1p(amp + self.eps)

    def hankel_bases(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features

        # x 为无量纲归一化距离 rho(x)=d(x)/L_ref
        rho = x
        kappa = self.current_kappa()

        # 与主模型完全一致：z(x)=max{kappa*rho(x), z_min}
        z = (kappa * rho).clamp_min(self.z_min)

        hankel = []
        for order in self.orders:
            phi = self._hankel_amp(order, z)
            hankel.append(phi.unsqueeze(-1))

        return torch.cat(hankel, dim=-1)  # (B, F, n_orders)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)

        hankel = self.hankel_bases(x_flat)

        # Dynamic sum over the selected Hankel orders.
        y = None
        for idx, order in enumerate(self.orders):
            order_out = F.linear(hankel[..., idx], self.hankel_weights[str(order)])
            y = order_out if y is None else (y + order_out)

        y = y.view(*original_shape[:-1], self.out_features)
        return y