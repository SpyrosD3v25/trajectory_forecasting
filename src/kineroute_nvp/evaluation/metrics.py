from __future__ import annotations

from typing import Dict

import torch

from kineroute_nvp.geometry.kinematics import build_sequence, sequence_kinematics
from kineroute_nvp.losses.trajectory_physics import ade, fde


