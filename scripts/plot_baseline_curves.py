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


def prettify_run_name(run_name: str) -> str:
    cleaned = run_name
    for suffix in ("_paper_clean", "_paper", "_coords", "_clean"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    aliases = {
        "seq2seq": "Seq2Seq",
        "gru": "GRU",
        "bigru": "Bi-GRU",
        "lstm": "LSTM",
        "bilstm": "Bi-LSTM",
        "kine_real_nvp": "Kine-Real-NVP",
    }
    return aliases.get(cleaned, cleaned.replace("_", " "))


def plot_metric(run_dirs: list[Path], metric_name: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for run_dir in run_dirs:
        epochs, ades, fdes = load_series(run_dir)
        series = ades if metric_name == "ADE" else fdes
        plt.plot(epochs, series, label=prettify_run_name(run_dir.name.split("__", 2)[-1]))
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Validation {metric_name} by Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    run_dirs = sorted(path for path in results_root.glob(f"*__{args.experiment_name}__*") if (path / "logs" / "metrics.jsonl").exists())
    output_dir = Path(args.output_dir)
    plot_metric(run_dirs, "ADE", output_dir / "val_ade_vs_epoch.png")
    plot_metric(run_dirs, "FDE", output_dir / "val_fde_vs_epoch.png")
