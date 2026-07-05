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


