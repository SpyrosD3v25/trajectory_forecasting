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


