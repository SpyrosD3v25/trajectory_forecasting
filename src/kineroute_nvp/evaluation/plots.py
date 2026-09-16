from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

# Training is a headless workload.  Selecting a GUI backend here is unsafe
# when DataLoader forks worker processes: inherited Tk state can abort a worker
# during garbage collection (Tcl_AsyncDelete).  This must happen before pyplot
# is imported.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def _atomic_save_figure(figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    figure.savefig(temporary, dpi=160, format="png")
    os.replace(temporary, output_path)


def _validation_series(records: list[dict], metric: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for record in records:
        validation = record.get("val", {})
        value = validation.get(metric, record.get(f"val_{metric}"))
        if value is not None:
            epochs.append(int(record["epoch"]))
            values.append(float(value))
    return epochs, values


def plot_validation_curves(records: list[dict], output_dir: Path) -> None:
    """Write the canonical validation ADE and FDE plots for a run."""
    for metric in ("ade", "fde"):
        epochs, values = _validation_series(records, metric)
        figure, axis = plt.subplots(figsize=(8, 5))
        if epochs:
            axis.plot(epochs, values, marker="o", markersize=2, label=f"val_{metric}")
            axis.legend()
        else:
            axis.text(0.5, 0.5, "No completed epochs", ha="center", va="center", transform=axis.transAxes)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.upper())
        axis.set_title(f"Validation {metric.upper()} vs Epoch")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        _atomic_save_figure(figure, Path(output_dir) / f"val_{metric}_vs_epoch.png")
        plt.close(figure)


def plot_validation_comparison(curves: list[dict[str, Any]], output_path: Path, title: str) -> None:
    """Plot multiple validation histories with one stable color per run."""
    if not curves:
        raise ValueError("At least one validation curve is required.")
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    color_map = plt.get_cmap("tab10")
    for index, curve in enumerate(curves):
        color = color_map(index % color_map.N)
        label = str(curve["label"])
        epochs = curve["epochs"]
        axes[0].plot(epochs, curve["ade"], color=color, linewidth=1.8, label=label)
        axes[1].plot(epochs, curve["fde"], color=color, linewidth=1.8, label=label)
    for axis, metric in zip(axes, ("ADE", "FDE")):
        axis.set_ylabel(f"Validation {metric}")
        axis.grid(True, alpha=0.3)
    axes[1].set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(title, y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(3, len(labels)),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    _atomic_save_figure(figure, Path(output_path))
    plt.close(figure)


def plot_loss_curves(metrics_jsonl: Path, output_path: Path) -> None:
    epochs = []
    train_total = []
    val_ade = []
    with metrics_jsonl.open() as handle:
        for line in handle:
            record = json.loads(line)
            epochs.append(record["epoch"])
            train_total.append(record["train"].get("total", record["train"].get("objective", 0.0)))
            val_ade.append(record["val"]["ade"])
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_total, label="train_total")
    ax1.plot(epochs, val_ade, label="val_ade")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_trajectory_examples(
    history,
    future,
    predicted,
    output_path: Path,
    max_examples: int,
    *,
    sample_ids: list[str] | None = None,
    seed: int = 42,
) -> list[int]:
    total = min(len(history), len(future), len(predicted))
    examples = min(total, max_examples)
    if examples <= 0:
        raise ValueError("At least one trajectory example is required.")
    indices = np.random.default_rng(seed).choice(total, size=examples, replace=False).tolist()
    cols = min(4, examples)
    rows = (examples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= examples:
            ax.axis("off")
            continue
        sample_idx = indices[idx]
        ax.plot(history[sample_idx, :, 0], history[sample_idx, :, 1], label="history", color="tab:blue")
        ax.plot(future[sample_idx, :, 0], future[sample_idx, :, 1], label="true_future", color="tab:green")
        ax.plot(predicted[sample_idx, :, 0], predicted[sample_idx, :, 1], label="pred_future", color="tab:red")
        ax.scatter(history[sample_idx, -1, 0], history[sample_idx, -1, 1], color="black", s=18)
        if sample_ids is not None:
            ax.set_title(str(sample_ids[sample_idx]), fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return indices
