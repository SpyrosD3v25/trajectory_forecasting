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
