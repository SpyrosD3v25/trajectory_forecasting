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
