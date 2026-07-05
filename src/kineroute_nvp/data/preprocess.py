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


