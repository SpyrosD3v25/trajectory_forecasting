from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def wrap_angle_numpy(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


