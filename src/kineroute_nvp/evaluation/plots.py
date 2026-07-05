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


