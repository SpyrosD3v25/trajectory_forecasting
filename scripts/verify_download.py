from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ALIASES = {
    "dma_clean": "paper_dma_clean_ship_core_lite_v1",
    "noaa_clean": "paper_noaa_clean_ship_core_lite_v1",
    "dma_lite": "paper_dma_ship_core_lite",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a downloaded EnvShip paper subset.")
    parser.add_argument("--dataset", choices=sorted(ALIASES), default="dma_clean")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    return parser.parse_args()


