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


