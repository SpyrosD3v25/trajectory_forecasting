from __future__ import annotations

import csv
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from kineroute_nvp.data.dataset import ProcessedTrajectoryDataset
from kineroute_nvp.data.preprocess import PrepareSummary, prepare_envship_dataset
from kineroute_nvp.evaluation.metrics import summarize_predictions
from kineroute_nvp.evaluation.plots import plot_loss_curves, plot_trajectory_examples
from kineroute_nvp.geometry.kinematics import decode_chart_torch
from kineroute_nvp.losses.trajectory_physics import ade, trajectory_loss
from kineroute_nvp.models.kineroute_nvp import KineRouteNVP
from kineroute_nvp.utils.io import append_jsonl, atomic_write_json, environment_snapshot, get_git_commit, write_json, write_yaml
from kineroute_nvp.utils.seed import set_seed


def _device_from_config(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


