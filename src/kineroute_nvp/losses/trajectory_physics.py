from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from kineroute_nvp.geometry.kinematics import build_sequence, sequence_kinematics


def ade(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(predicted - target, dim=-1).mean()


