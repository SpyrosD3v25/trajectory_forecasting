from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


