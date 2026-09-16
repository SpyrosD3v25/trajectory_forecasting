from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

PreprocessingMode = Literal["raw", "heading_aligned", "heading_aligned_displacement"]


def _validate_mode(mode: str) -> None:
    if mode not in {"raw", "heading_aligned", "heading_aligned_displacement"}:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")


def _as_trajectory(trajectory: np.ndarray) -> np.ndarray:
    array = np.asarray(trajectory, dtype=np.float64)
    if array.shape != (30, 2):
        raise ValueError(f"Expected trajectory with shape (30, 2), got {array.shape}")
    return array


def _last_heading(trajectory: np.ndarray, eps: float = 1e-8) -> float:
    displacements = np.diff(trajectory, axis=0)
    norms = np.linalg.norm(displacements, axis=-1)
    valid = np.where(norms > eps)[0]
    if len(valid) == 0:
        return 0.0
    dx, dy = displacements[valid[-1]]
    return float(np.arctan2(dy, dx))


def _rotation(angle: float) -> np.ndarray:
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)


@dataclass(frozen=True)
class TrajectoryPreprocessor:
    mode: PreprocessingMode
    origin: np.ndarray
    heading: float

    @classmethod
    def fit(cls, trajectory: np.ndarray, mode: PreprocessingMode) -> "TrajectoryPreprocessor":
        reference = _as_trajectory(trajectory)
        _validate_mode(mode)
        heading = 0.0 if mode == "raw" else _last_heading(reference)
        # Alignment is anchored at the final observed point.  Using the first
        # point would change both translation and rotation in the alignment
        # ablation instead of isolating rotation.
        return cls(mode=mode, origin=reference[-1].copy(), heading=heading)

    def encode(self, trajectory: np.ndarray) -> np.ndarray:
        positions = _as_trajectory(trajectory)
        if self.mode == "raw":
            return positions.astype(np.float32)

        aligned = (positions - self.origin) @ _rotation(-self.heading).T
        if self.mode == "heading_aligned":
            return aligned.astype(np.float32)

        displacements = np.zeros_like(aligned)
        displacements[0] = aligned[0]
        displacements[1:] = aligned[1:] - aligned[:-1]
        return displacements.astype(np.float32)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        values = _as_trajectory(encoded)
        if self.mode == "raw":
            return values.astype(np.float32)

        if self.mode == "heading_aligned":
            aligned = values
        else:
            aligned = np.cumsum(values, axis=0)
        positions = aligned @ _rotation(self.heading).T + self.origin
        return positions.astype(np.float32)


def encode_trajectory(trajectory: np.ndarray, mode: PreprocessingMode) -> tuple[np.ndarray, TrajectoryPreprocessor]:
    transform = TrajectoryPreprocessor.fit(trajectory, mode)
    return transform.encode(trajectory), transform


def decode_trajectory(encoded: np.ndarray, transform: TrajectoryPreprocessor) -> np.ndarray:
    return transform.decode(encoded)


def encode_history_future(
    history: np.ndarray,
    future: np.ndarray,
    mode: PreprocessingMode,
) -> tuple[np.ndarray, np.ndarray, TrajectoryPreprocessor]:
    transform = TrajectoryPreprocessor.fit(history, mode)
    return transform.encode(history), transform.encode(future), transform


def _as_batched_torch_trajectory(trajectory: torch.Tensor, name: str) -> torch.Tensor:
    if trajectory.ndim != 3 or trajectory.shape[-1] != 2:
        raise ValueError(f"Expected {name} shape [B, T, 2], got {tuple(trajectory.shape)}")
    return trajectory


def _last_heading_torch(history: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    displacement = history[:, 1:, :] - history[:, :-1, :]
    speed = torch.linalg.norm(displacement, dim=-1)
    raw_heading = torch.atan2(displacement[..., 1], displacement[..., 0])
    heading = torch.zeros(history.shape[0], device=history.device, dtype=history.dtype)
    for step in range(displacement.shape[1]):
        heading = torch.where(speed[:, step] > eps, raw_heading[:, step], heading)
    return heading


def _rotate_torch(points: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    cos_a = torch.cos(angle)[:, None]
    sin_a = torch.sin(angle)[:, None]
    x = points[..., 0]
    y = points[..., 1]
    return torch.stack([x * cos_a - y * sin_a, x * sin_a + y * cos_a], dim=-1)


def encode_with_history_torch(history: torch.Tensor, trajectory: torch.Tensor, mode: PreprocessingMode) -> torch.Tensor:
    """Encode a batched trajectory using the origin and final heading of its history."""
    _validate_mode(mode)
    history = _as_batched_torch_trajectory(history, "history")
    trajectory = _as_batched_torch_trajectory(trajectory, "trajectory")
    if history.shape[0] != trajectory.shape[0]:
        raise ValueError("history and trajectory must have the same batch size")
    if mode == "raw":
        return trajectory

    aligned = _rotate_torch(trajectory - history[:, -1:, :], -_last_heading_torch(history))
    if mode == "heading_aligned":
        return aligned

    displacement = torch.zeros_like(aligned)
    displacement[:, 0, :] = aligned[:, 0, :]
    displacement[:, 1:, :] = aligned[:, 1:, :] - aligned[:, :-1, :]
    return displacement


def decode_with_history_torch(history: torch.Tensor, encoded: torch.Tensor, mode: PreprocessingMode) -> torch.Tensor:
    """Decode a batched representation using the same history-derived transform."""
    _validate_mode(mode)
    history = _as_batched_torch_trajectory(history, "history")
    encoded = _as_batched_torch_trajectory(encoded, "encoded trajectory")
    if history.shape[0] != encoded.shape[0]:
        raise ValueError("history and encoded trajectory must have the same batch size")
    if mode == "raw":
        return encoded

    aligned = torch.cumsum(encoded, dim=1) if mode == "heading_aligned_displacement" else encoded
    return _rotate_torch(aligned, _last_heading_torch(history)) + history[:, -1:, :]
