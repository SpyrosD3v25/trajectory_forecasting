from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "mark000071/EnvShip-Bench_An_Environment-Enhanced_Benchmark_for_Short-Term_Vessel_Trajectory_Prediction"
OUT_DIR = Path("data/envship")

DATASETS = {
    "dma_clean": {
        "remote_root": "DMA/mini_benchmark/clean_ship_core_lite_v1",
        "alias_root": "paper_dma_clean_ship_core_lite_v1",
    },
    "noaa_clean": {
        "remote_root": "NOAA/mini_bench/clean_ship_core_lite_v1",
        "alias_root": "paper_noaa_clean_ship_core_lite_v1",
    },
    "dma_lite": {
        "remote_root": "DMA/mini_benchmark/ship_core_lite",
        "alias_root": "paper_dma_ship_core_lite",
    },
}

DOC_PATTERNS = ["README.md"]


