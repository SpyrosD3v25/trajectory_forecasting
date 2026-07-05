from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


CONFIG_TO_ARG = {
    "run_name": "run_name",
    "seed": "seed",
    "data.raw_root": "data_root",
    "data.raw_roots": "raw_roots",
    "data.processed_root": "processed_root",
    "data.train_split": "train_split",
    "data.val_split": "val_split",
    "data.test_split": "test_split",
    "data.dt_seconds": "dt_seconds",
    "data.history_steps": "history_steps",
    "data.future_steps": "future_steps",
    "data.prepare_if_missing": "prepare_if_missing",
    "data.align_to_last_heading": "align_to_last_heading",
    "data.chart_type": "chart_type",
    "data.max_train_samples": "max_train_samples",
    "data.max_val_samples": "max_val_samples",
    "data.max_test_samples": "max_test_samples",
    "model.name": "model_name",
    "model.num_routed_blocks": "num_routed_blocks",
    "model.hidden_dim": "hidden_dim",
    "model.hidden_layers": "hidden_layers",
    "model.scale_bound": "scale_bound",
    "loss.lambda_velocity": "lambda_velocity",
    "loss.lambda_acceleration": "lambda_acceleration",
    "loss.lambda_turn": "lambda_turn",
    "loss.lambda_feasibility": "lambda_feasibility",
    "loss.lambda_reverse": "lambda_reverse",
    "loss.lambda_position_mse": "lambda_position_mse",
    "loss.lambda_fde": "lambda_fde",
    "loss.feasibility_quantile": "feasibility_quantile",
    "optimizer.name": "optimizer_name",
    "optimizer.learning_rate": "learning_rate",
    "optimizer.weight_decay": "weight_decay",
    "scheduler.name": "scheduler_name",
    "scheduler.factor": "scheduler_factor",
    "scheduler.patience": "scheduler_patience",
    "training.epochs": "epochs",
    "training.batch_size": "batch_size",
    "training.num_workers": "num_workers",
    "training.checkpoint_every_epochs": "checkpoint_every_epochs",
    "training.early_stopping_patience": "early_stopping_patience",
    "training.device": "device",
    "evaluation.save_prediction_examples": "save_prediction_examples",
    "evaluation.evaluate_reverse_mapping": "evaluate_reverse_mapping",
    "output.results_root": "results_root",
}


def _flatten(data: dict, prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, path))
        else:
            flattened[path] = value
    return flattened


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


