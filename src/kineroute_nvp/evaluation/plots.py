from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curves(metrics_jsonl: Path, output_path: Path) -> None:
    epochs = []
    train_total = []
    val_ade = []
    with metrics_jsonl.open() as handle:
        for line in handle:
            record = json.loads(line)
            epochs.append(record["epoch"])
            train_total.append(record["train"]["total"])
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


def plot_trajectory_examples(history, future, predicted, output_path: Path, max_examples: int) -> None:
    examples = min(len(history), max_examples)
    cols = min(4, examples)
    rows = (examples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= examples:
            ax.axis("off")
            continue
        ax.plot(history[idx, :, 0], history[idx, :, 1], label="history", color="tab:blue")
        ax.plot(future[idx, :, 0], future[idx, :, 1], label="true_future", color="tab:green")
        ax.plot(predicted[idx, :, 0], predicted[idx, :, 1], label="pred_future", color="tab:red")
        ax.scatter(history[idx, -1, 0], history[idx, -1, 1], color="black", s=18)
        if idx == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
