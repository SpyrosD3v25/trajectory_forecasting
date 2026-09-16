from __future__ import annotations

import math

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
        _, hidden = self.encoder(history)
        if self.bidirectional:
            # The backward state at encoded[:, -1] has only seen the final
            # input.  The two terminal hidden states each summarize their full
            # direction and are the correct bidirectional sequence encoding.
            final = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            final = hidden[-1]
        output = self.head(final)
        return output.view(history.shape[0], 30, 2)


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(
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
        _, (hidden, _) = self.encoder(history)
        if self.bidirectional:
            final = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            final = hidden[-1]
        output = self.head(final)
        return output.view(history.shape[0], 30, 2)


class Seq2SeqBaseline(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, history: torch.Tensor, target_future: torch.Tensor | None = None, teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        _, hidden = self.encoder(history)
        decoder_input = history[:, -1:, :]
        outputs = []
        for step_idx in range(30):
            decoded, hidden = self.decoder(decoder_input, hidden)
            step = self.output(decoded)
            outputs.append(step)
            if self.training and target_future is not None and teacher_forcing_ratio > 0.0 and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_future[:, step_idx : step_idx + 1, :]
            else:
                decoder_input = step
        return torch.cat(outputs, dim=1)


class DeadReckoningBaseline(nn.Module):
    def __init__(self, velocity_points: int = 2, future_steps: int = 30) -> None:
        super().__init__()
        if velocity_points < 2:
            raise ValueError("velocity_points must be at least 2")
        self.velocity_points = velocity_points
        self.future_steps = future_steps

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[-2:] != (30, 2):
            raise ValueError(f"Expected history shape [B, 30, 2], got {tuple(history.shape)}")
        points = min(self.velocity_points, history.shape[1])
        window = history[:, -points:, :]
        velocity = (window[:, -1, :] - window[:, 0, :]) / float(points - 1)
        steps = torch.arange(1, self.future_steps + 1, device=history.device, dtype=history.dtype)
        return history[:, -1:, :] + steps.view(1, -1, 1) * velocity.unsqueeze(1)


class MLPForecaster(nn.Module):
    def __init__(
        self,
        input_steps: int = 30,
        future_steps: int = 30,
        input_dim: int = 2,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        current_dim = input_steps * input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, future_steps * input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        output = self.net(history.flatten(start_dim=1))
        return output.view(history.shape[0], self.future_steps, self.input_dim)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ResidualMLPForecaster(nn.Module):
    def __init__(
        self,
        input_steps: int = 30,
        future_steps: int = 30,
        input_dim: int = 2,
        hidden_dim: int = 256,
        residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.input_dim = input_dim
        blocks: list[nn.Module] = [nn.Linear(input_steps * input_dim, hidden_dim), nn.ReLU()]
        blocks.extend(ResidualBlock(hidden_dim) for _ in range(residual_blocks))
        blocks.append(nn.Linear(hidden_dim, future_steps * input_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        output = self.net(history.flatten(start_dim=1))
        return output.view(history.shape[0], self.future_steps, self.input_dim)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        levels: int = 4,
        kernel_size: int = 3,
        future_steps: int = 30,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        layers: list[nn.Module] = []
        channels = input_dim
        for level in range(levels):
            dilation = 2**level
            padding = (kernel_size - 1) * dilation
            layers.extend(
                [
                    nn.Conv1d(channels, hidden_dim, kernel_size, padding=padding, dilation=dilation),
                    nn.ReLU(),
                ]
            )
            channels = hidden_dim
        self.temporal = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, future_steps * input_dim)
        self.input_dim = input_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        features = history.transpose(1, 2)
        encoded = self.temporal(features)[..., : history.shape[1]]
        final = encoded[..., -1]
        output = self.head(final)
        return output.view(history.shape[0], self.future_steps, self.input_dim)


class TransformerNARForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        future_steps: int = 30,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, 30, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, future_steps * input_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        positions = torch.arange(30, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, self.position.shape[-1], 2, dtype=torch.float32) * (-math.log(10000.0) / self.position.shape[-1]))
        encoding = torch.zeros(30, self.position.shape[-1])
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        with torch.no_grad():
            self.position.copy_(encoding.unsqueeze(0))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(history) + self.position[:, : history.shape[1], :])
        pooled = encoded.mean(dim=1)
        output = self.head(pooled)
        return output.view(history.shape[0], self.future_steps, self.input_dim)
