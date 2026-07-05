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


def load_series(run_dir: Path) -> tuple[list[int], list[float], list[float]]:
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    epochs: list[int] = []
    ades: list[float] = []
    fdes: list[float] = []
    with metrics_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            epochs.append(int(record["epoch"]))
            val = record["val"]
            ades.append(float(val["ade"]))
            fdes.append(float(val["fde"]))
    return epochs, ades, fdes


