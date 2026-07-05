from __future__ import annotations

import torch
from torch import nn

from .coupling import RoutedCouplingBlock


class KineRouteNVP(nn.Module):
    def __init__(self, num_routed_blocks: int, hidden_dim: int, hidden_layers: int, scale_bound: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [RoutedCouplingBlock(hidden_dim=hidden_dim, hidden_layers=hidden_layers, scale_bound=scale_bound) for _ in range(num_routed_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y
