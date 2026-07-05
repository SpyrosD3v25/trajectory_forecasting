from __future__ import annotations

import torch
from torch import nn


class GRUBaseline(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 30 * 2),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        encoded, hidden = self.encoder(history)
        if self.bidirectional:
            final = encoded[:, -1, :]
        else:
            final = hidden[-1]
        output = self.head(final)
        return output.view(history.shape[0], 30, 2)


