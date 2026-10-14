# KANLinear_angular.py
# Complete Tx-centered angular Hankel basis for the angular comparison experiment.

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
class AngularHankelKANLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_order: int = 1,
        learnable_k: bool = True,
        z_min: float = math.pi * 1e-3,
        kappa_min: float = 0.1,
        kappa_max: float = 10.0,
        kappa_init: float = math.pi,
        eps: float = 1e-6,
        base_activation=nn.SiLU,
        scale_base: float = 1.0,
        scale_hankel: float = 1.0,
        r_min=None,  # compatibility only; not used by the finalized formulation
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.max_order = int(max_order)

        if self.max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {self.max_order}")

        self.orders = tuple(range(self.max_order + 1))

        self.z_min = float(z_min)
        self.eps = float(eps)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.kappa_init = float(kappa_init)

        if self.z_min <= 0:
            raise ValueError(f"z_min must be positive, got {self.z_min}")
        if self.kappa_min <= 0:
            raise ValueError(f"kappa_min must be positive, got {self.kappa_min}")
        if self.kappa_max <= self.kappa_min:
            raise ValueError(
                f"kappa_max must be greater than kappa_min, got "
                f"[{self.kappa_min}, {self.kappa_max}]"
            )
        if not (self.kappa_min < self.kappa_init < self.kappa_max):
            raise ValueError(
                f"kappa_init must lie strictly inside "
                f"({self.kappa_min}, {self.kappa_max}), got {self.kappa_init}"
            )

        # Kept only for interface compatibility with the radial implementation.
        # No separate base branch is used in the finalized model.
        self.base_activation = base_activation()
        self.scale_base = float(scale_base)
        self.scale_hankel = float(scale_hankel)

        # Bounded kappa:
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

        # Two independent learnable weight matrices (Re and Im) per order.
        # There is intentionally no unused base_weight.
        self.real_weights = nn.ParameterDict({
            str(n): nn.Parameter(torch.empty(out_features, in_features))
            for n in self.orders
        })
        self.imag_weights = nn.ParameterDict({
            str(n): nn.Parameter(torch.empty(out_features, in_features))
            for n in self.orders
        })

        self.reset_parameters()

    def reset_parameters(self):
        for n in self.orders:
            w_re = self.real_weights[str(n)]
            w_im = self.imag_weights[str(n)]

            nn.init.kaiming_uniform_(
                w_re,
                a=math.sqrt(5) * self.scale_hankel,
            )
            nn.init.kaiming_uniform_(
                w_im,
                a=math.sqrt(5) * self.scale_hankel,
            )

            with torch.no_grad():
                w_re.mul_(self.scale_hankel)
                w_im.mul_(self.scale_hankel)

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

    def _signed_log1p(self, x: torch.Tensor) -> torch.Tensor:
        """Sign-preserving dynamic-range compression."""
        return torch.sign(x) * torch.log1p(torch.abs(x) + self.eps)

    def angular_bases(
        self,
        rho: torch.Tensor,
        phi: torch.Tensor,
    ):
        """
        rho, phi : (B, F)

        rho is the fixed-L_ref normalized Tx distance, and phi is the
        Tx-centered azimuth in radians.
        """
        if rho.dim() != 2 or rho.size(1) != self.in_features:
            raise ValueError(
                f"rho must have shape (B, {self.in_features}), "
                f"got {tuple(rho.shape)}"
            )

        if phi.shape != rho.shape:
            raise ValueError(
                f"phi must have the same shape as rho, got "
                f"rho={tuple(rho.shape)}, phi={tuple(phi.shape)}"
            )

        # Finalized radial-coordinate handling:
        # rho is non-negative and dimensionless; kappa is positive by construction.
        kappa = self.current_kappa()
        z = (kappa * rho).clamp_min(self.z_min)

        bases = []

        for n in self.orders:
            J, Y = self._bessel_jy_integer_order(n, z)

            angle = float(n) * phi
            cos_n = torch.cos(angle)
            sin_n = torch.sin(angle)

            re = J * cos_n - Y * sin_n
            im = J * sin_n + Y * cos_n

            re = self._signed_log1p(re)
            im = self._signed_log1p(im)

            bases.append((re, im))

        return bases

    def forward(
        self,
        rho: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        if rho.size(-1) != self.in_features:
            raise ValueError(
                f"Last dimension of rho must equal {self.in_features}, "
                f"got {rho.size(-1)}"
            )

        if phi.shape != rho.shape:
            raise ValueError(
                f"phi must have the same shape as rho, got "
                f"rho={tuple(rho.shape)}, phi={tuple(phi.shape)}"
            )

        original_shape = rho.shape

        rho_flat = rho.reshape(-1, self.in_features)
        phi_flat = phi.reshape(-1, self.in_features)

        bases = self.angular_bases(rho_flat, phi_flat)

        y = None
        for n, (re, im) in zip(self.orders, bases):
            out_re = F.linear(re, self.real_weights[str(n)])
            out_im = F.linear(im, self.imag_weights[str(n)])
            out_n = out_re + out_im

            y = out_n if y is None else (y + out_n)

        return y.view(*original_shape[:-1], self.out_features)