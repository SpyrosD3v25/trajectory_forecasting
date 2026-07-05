from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot validation ADE/FDE curves across compact-benchmark runs.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--experiment-name", default="baselines")
    parser.add_argument("--output-dir", default="results/baseline_plots")
    return parser.parse_args()


