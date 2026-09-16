from __future__ import annotations

import torch
from torch import nn


def _make_mlp(in_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    layers = []
    current = in_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(nn.ReLU())
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    net = nn.Sequential(*layers)
    last = net[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return net


class AffineSubnet(nn.Module):
    def __init__(self, in_dim: int, target_dim: int, hidden_dim: int, hidden_layers: int, scale_bound: float) -> None:
        super().__init__()
        self.scale_bound = scale_bound
        self.scale_net = _make_mlp(in_dim, target_dim, hidden_dim, hidden_layers)
        self.shift_net = _make_mlp(in_dim, target_dim, hidden_dim, hidden_layers)

    def forward(self, conditioning: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale = self.scale_bound * torch.tanh(self.scale_net(conditioning))
        shift = self.shift_net(conditioning)
        return log_scale, shift


class RoutedCouplingBlock(nn.Module):
    def __init__(self, hidden_dim: int, hidden_layers: int, scale_bound: float) -> None:
        super().__init__()
        self.t_affine = AffineSubnet(31, 29, hidden_dim, hidden_layers, scale_bound)
        self.s_affine = AffineSubnet(31, 29, hidden_dim, hidden_layers, scale_bound)
        self.a_affine = AffineSubnet(58, 2, hidden_dim, hidden_layers, scale_bound)

    @staticmethod
    def _split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return x[:, :2], x[:, 2:31], x[:, 31:]

    @staticmethod
    def _merge(a: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, s, t], dim=-1)

    @staticmethod
    def _forward_affine(x: torch.Tensor, log_scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x * torch.exp(log_scale) + shift

    @staticmethod
    def _inverse_affine(y: torch.Tensor, log_scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return (y - shift) * torch.exp(-log_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, s, t = self._split(x)
        t_scale, t_shift = self.t_affine(torch.cat([a, s], dim=-1))
        t_prime = self._forward_affine(t, t_scale, t_shift)
        s_scale, s_shift = self.s_affine(torch.cat([a, t_prime], dim=-1))
        s_prime = self._forward_affine(s, s_scale, s_shift)
        a_scale, a_shift = self.a_affine(torch.cat([s_prime, t_prime], dim=-1))
        a_prime = self._forward_affine(a, a_scale, a_shift)
        return self._merge(a_prime, s_prime, t_prime)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        a_prime, s_prime, t_prime = self._split(y)
        a_scale, a_shift = self.a_affine(torch.cat([s_prime, t_prime], dim=-1))
        a = self._inverse_affine(a_prime, a_scale, a_shift)
        s_scale, s_shift = self.s_affine(torch.cat([a, t_prime], dim=-1))
        s = self._inverse_affine(s_prime, s_scale, s_shift)
        t_scale, t_shift = self.t_affine(torch.cat([a, s], dim=-1))
        t = self._inverse_affine(t_prime, t_scale, t_shift)
        return self._merge(a, s, t)


class StandardCouplingBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_layers: int, scale_bound: float, flip: bool = False) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be at least 2")
        self.dim = dim
        self.flip = flip
        self.left_dim = dim // 2
        self.right_dim = dim - self.left_dim
        self.affine = AffineSubnet(self.left_dim, self.right_dim, hidden_dim, hidden_layers, scale_bound)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flip:
            x = torch.flip(x, dims=(-1,))
        left = x[:, : self.left_dim]
        right = x[:, self.left_dim :]
        return left, right

    def _merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        output = torch.cat([left, right], dim=-1)
        if self.flip:
            output = torch.flip(output, dims=(-1,))
        return output

    @staticmethod
    def _forward_affine(x: torch.Tensor, log_scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x * torch.exp(log_scale) + shift

    @staticmethod
    def _inverse_affine(y: torch.Tensor, log_scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return (y - shift) * torch.exp(-log_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self._split(x)
        scale, shift = self.affine(left)
        return self._merge(left, self._forward_affine(right, scale, shift))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        left, right = self._split(y)
        scale, shift = self.affine(left)
        return self._merge(left, self._inverse_affine(right, scale, shift))


class NonInvertibleRoutedBlock(nn.Module):
    def __init__(self, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        self.t_net = _make_mlp(60, 29, hidden_dim, hidden_layers)
        self.s_net = _make_mlp(60, 29, hidden_dim, hidden_layers)
        self.a_net = _make_mlp(60, 2, hidden_dim, hidden_layers)

    @staticmethod
    def _split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return x[:, :2], x[:, 2:31], x[:, 31:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, s, t = self._split(x)
        full = torch.cat([a, s, t], dim=-1)
        # Every update sees its own target values, so these are unconstrained
        # residual maps rather than analytically invertible triangular couplings.
        t_prime = t + self.t_net(full)
        s_prime = s + self.s_net(full)
        a_prime = a + self.a_net(full)
        return torch.cat([a_prime, s_prime, t_prime], dim=-1)
