from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from kineroute_nvp.geometry.kinematics import (
    describe_chart_choice,
    describe_displacement_chart_choice,
    describe_position_chart_choice,
    encode_positions_numpy,
)


REQUIRED_COLUMNS = ["sample_id", "hist_x_json", "hist_y_json", "fut_x_json", "fut_y_json"]


@dataclass
class PrepareSummary:
    processed_root: Path
    metadata_path: Path
    normalization_path: Path
    thresholds_path: Path
    split_sizes: Dict[str, int]


def _parse_positions(row: pd.Series, prefix: str) -> np.ndarray:
    x = np.asarray(json.loads(row[f"{prefix}_x_json"]), dtype=np.float64)
    y = np.asarray(json.loads(row[f"{prefix}_y_json"]), dtype=np.float64)
    if x.shape != (30,) or y.shape != (30,):
        raise ValueError(f"{prefix} arrays must be length 30, got {x.shape} and {y.shape}")
    return np.stack([x, y], axis=-1)


def _compute_thresholds(train_future: np.ndarray, train_history: np.ndarray, dt_seconds: float, quantile: float) -> Dict[str, float]:
    last_history = train_history[:, -1:, :]
    sequence = np.concatenate([last_history, train_future], axis=1)
    velocity = np.diff(sequence, axis=1) / dt_seconds
    acceleration = np.diff(velocity, axis=1) / dt_seconds
    speed = np.linalg.norm(velocity, axis=-1)
    heading = np.zeros_like(speed)
    raw_heading = np.arctan2(velocity[..., 1], velocity[..., 0])
    heading[:, 0] = np.where(speed[:, 0] > 1e-8, raw_heading[:, 0], 0.0)
    for idx in range(1, speed.shape[1]):
        heading[:, idx] = np.where(speed[:, idx] > 1e-8, raw_heading[:, idx], heading[:, idx - 1])
    turn_rate = np.arctan2(np.sin(heading[:, 1:] - heading[:, :-1]), np.cos(heading[:, 1:] - heading[:, :-1])) / dt_seconds
    return {
        "max_acceleration": float(np.quantile(np.linalg.norm(acceleration, axis=-1), quantile)),
        "max_turn_rate": float(np.quantile(np.abs(turn_rate), quantile)),
    }


def _align_pair_to_last_heading(history: np.ndarray, future: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    displacements = np.diff(history, axis=0)
    norms = np.linalg.norm(displacements, axis=-1)
    valid = np.where(norms > eps)[0]
    if len(valid) == 0:
        angle = 0.0
    else:
        last_disp = displacements[valid[-1]]
        angle = np.arctan2(last_disp[1], last_disp[0])
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return history @ rotation.T, future @ rotation.T


