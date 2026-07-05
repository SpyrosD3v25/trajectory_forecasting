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


