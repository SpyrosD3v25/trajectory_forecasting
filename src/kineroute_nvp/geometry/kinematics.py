from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def wrap_angle_numpy(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def positions_to_chart_torch(positions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if positions.shape[-2:] != (30, 2):
        raise ValueError(f"Expected positions[..., 30, 2], got {tuple(positions.shape)}")
    displacements = positions[..., 1:, :] - positions[..., :-1, :]
    radii = torch.linalg.norm(displacements, dim=-1)
    raw_headings = torch.atan2(displacements[..., 1], displacements[..., 0])
    headings = torch.zeros_like(radii)
    headings[..., 0] = torch.where(radii[..., 0] > eps, raw_headings[..., 0], torch.zeros_like(raw_headings[..., 0]))
    for idx in range(1, radii.shape[-1]):
        headings[..., idx] = torch.where(radii[..., idx] > eps, raw_headings[..., idx], headings[..., idx - 1])
    delta = wrap_angle_torch(headings[..., 1:] - headings[..., :-1])
    return torch.cat([positions[..., 0, :], radii, headings[..., :1], delta], dim=-1)


def chart_to_positions_torch(chart: torch.Tensor) -> torch.Tensor:
    if chart.shape[-1] != 60:
        raise ValueError(f"Expected chart[..., 60], got {tuple(chart.shape)}")
    anchor = chart[..., :2]
    radii = chart[..., 2:31]
    heading_2 = chart[..., 31:32]
    delta = chart[..., 32:]
    headings = torch.cat([heading_2, heading_2 + torch.cumsum(delta, dim=-1)], dim=-1)
    displacements = torch.stack([radii * torch.cos(headings), radii * torch.sin(headings)], dim=-1)
    cumulative = torch.cumsum(displacements, dim=-2)
    anchors = anchor.unsqueeze(-2)
    return torch.cat([anchors, anchors + cumulative], dim=-2)


def positions_to_displacement_chart_torch(positions: torch.Tensor) -> torch.Tensor:
    if positions.shape[-2:] != (30, 2):
        raise ValueError(f"Expected positions[..., 30, 2], got {tuple(positions.shape)}")
    displacements = positions[..., 1:, :] - positions[..., :-1, :]
    return torch.cat([positions[..., 0, :], displacements[..., 0], displacements[..., 1]], dim=-1)


def displacement_chart_to_positions_torch(chart: torch.Tensor) -> torch.Tensor:
    if chart.shape[-1] != 60:
        raise ValueError(f"Expected chart[..., 60], got {tuple(chart.shape)}")
    anchor = chart[..., :2]
    dx = chart[..., 2:31]
    dy = chart[..., 31:]
    displacements = torch.stack([dx, dy], dim=-1)
    cumulative = torch.cumsum(displacements, dim=-2)
    anchors = anchor.unsqueeze(-2)
    return torch.cat([anchors, anchors + cumulative], dim=-2)


def positions_to_position_chart_torch(positions: torch.Tensor) -> torch.Tensor:
    if positions.shape[-2:] != (30, 2):
        raise ValueError(f"Expected positions[..., 30, 2], got {tuple(positions.shape)}")
    anchor = positions[..., 0, :]
    forward = positions[..., 1:, 0]
    lateral = positions[..., 1:, 1]
    return torch.cat([anchor, forward, lateral], dim=-1)


def position_chart_to_positions_torch(chart: torch.Tensor) -> torch.Tensor:
    if chart.shape[-1] != 60:
        raise ValueError(f"Expected chart[..., 60], got {tuple(chart.shape)}")
    anchor = chart[..., :2]
    forward = chart[..., 2:31]
    lateral = chart[..., 31:]
    rest = torch.stack([forward, lateral], dim=-1)
    return torch.cat([anchor.unsqueeze(-2), rest], dim=-2)


def positions_to_chart_numpy(positions: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    tensor = torch.as_tensor(positions, dtype=torch.float64)
    return positions_to_chart_torch(tensor, eps=eps).cpu().numpy()


