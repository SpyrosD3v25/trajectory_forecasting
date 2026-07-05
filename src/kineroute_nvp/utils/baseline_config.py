from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


CONFIG_TO_ARG = {
    "run_name": "run_name",
    "seed": "seed",
    "data.raw_root": "raw_root",
    "data.raw_roots": "raw_roots",
    "data.processed_root": "processed_root",
    "data.include_ship_classes": "include_ship_classes",
    "data.quality_tiers": "quality_tiers",
    "data.max_train_samples": "max_train_samples",
    "data.max_val_samples": "max_val_samples",
    "data.max_test_samples": "max_test_samples",
    "model.name": "model_name",
    "model.hidden_dim": "hidden_dim",
    "model.num_layers": "num_layers",
    "loss.name": "loss_name",
    "loss.normalize": "normalize",
    "loss.teacher_forcing_ratio": "teacher_forcing_ratio",
    "optimizer.learning_rate": "learning_rate",
    "optimizer.weight_decay": "weight_decay",
    "training.epochs": "epochs",
    "training.batch_size": "batch_size",
    "training.num_workers": "num_workers",
    "training.device": "device",
    "output.results_root": "results_root",
}


