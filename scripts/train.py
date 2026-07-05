from __future__ import annotations

import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kineroute_nvp.training.trainer import ExperimentRunner
from kineroute_nvp.utils.config import args_to_config, parse_args


