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


def build_parser(defaults: Dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--experiment-name", default=defaults.get("experiment_name"))
    parser.add_argument("--run-name", default=defaults.get("run_name"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--data-root", default=defaults.get("data_root"))
    parser.add_argument("--raw-roots", nargs="*", default=defaults.get("raw_roots"))
    parser.add_argument("--processed-root", default=defaults.get("processed_root"))
    parser.add_argument("--train-split", default=defaults.get("train_split", "train"))
    parser.add_argument("--val-split", default=defaults.get("val_split", "val"))
    parser.add_argument("--test-split", default=defaults.get("test_split", "test"))
    parser.add_argument("--dt-seconds", type=float, default=defaults.get("dt_seconds", 20.0))
    parser.add_argument("--history-steps", type=int, default=defaults.get("history_steps", 30))
    parser.add_argument("--future-steps", type=int, default=defaults.get("future_steps", 30))
    parser.add_argument("--prepare-if-missing", type=_str2bool, default=defaults.get("prepare_if_missing", True))
    parser.add_argument("--align-to-last-heading", type=_str2bool, default=defaults.get("align_to_last_heading", False))
    parser.add_argument("--chart-type", choices=["polar", "displacement", "position"], default=defaults.get("chart_type", "polar"))
    parser.add_argument("--max-train-samples", type=int, default=defaults.get("max_train_samples"))
    parser.add_argument("--max-val-samples", type=int, default=defaults.get("max_val_samples"))
    parser.add_argument("--max-test-samples", type=int, default=defaults.get("max_test_samples"))
    parser.add_argument("--model-name", default=defaults.get("model_name", "kineroute_nvp"))
    parser.add_argument("--num-routed-blocks", type=int, default=defaults.get("num_routed_blocks", 6))
    parser.add_argument("--hidden-dim", type=int, default=defaults.get("hidden_dim", 256))
    parser.add_argument("--hidden-layers", type=int, default=defaults.get("hidden_layers", 2))
    parser.add_argument("--scale-bound", type=float, default=defaults.get("scale_bound", 1.5))
    parser.add_argument("--lambda-velocity", type=float, default=defaults.get("lambda_velocity", 0.10))
    parser.add_argument("--lambda-acceleration", type=float, default=defaults.get("lambda_acceleration", 0.05))
    parser.add_argument("--lambda-turn", type=float, default=defaults.get("lambda_turn", 0.05))
    parser.add_argument("--lambda-feasibility", type=float, default=defaults.get("lambda_feasibility", 0.01))
    parser.add_argument("--lambda-reverse", type=float, default=defaults.get("lambda_reverse", 0.0))
    parser.add_argument("--lambda-position-mse", type=float, default=defaults.get("lambda_position_mse", 0.0))
    parser.add_argument("--lambda-fde", type=float, default=defaults.get("lambda_fde", 0.0))
    parser.add_argument("--feasibility-quantile", type=float, default=defaults.get("feasibility_quantile", 0.995))
    parser.add_argument("--optimizer-name", default=defaults.get("optimizer_name", "adamw"))
    parser.add_argument("--learning-rate", type=float, default=defaults.get("learning_rate", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 1e-4))
    parser.add_argument("--scheduler-name", default=defaults.get("scheduler_name", "reduce_on_plateau"))
    parser.add_argument("--scheduler-factor", type=float, default=defaults.get("scheduler_factor", 0.5))
    parser.add_argument("--scheduler-patience", type=int, default=defaults.get("scheduler_patience", 8))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 80))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 256))
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 4))
    parser.add_argument("--checkpoint-every-epochs", type=int, default=defaults.get("checkpoint_every_epochs", 10))
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.get("early_stopping_patience", 15))
    parser.add_argument("--device", default=defaults.get("device", "auto"))
    parser.add_argument("--save-prediction-examples", type=int, default=defaults.get("save_prediction_examples", 24))
    parser.add_argument("--evaluate-reverse-mapping", type=_str2bool, default=defaults.get("evaluate_reverse_mapping", True))
    parser.add_argument("--results-root", default=defaults.get("results_root", "results"))
    return parser


def load_yaml_defaults(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    payload = yaml.safe_load(Path(config_path).read_text()) or {}
    flat = _flatten(payload)
    defaults = {}
    for key, value in flat.items():
        if key in CONFIG_TO_ARG:
            defaults[CONFIG_TO_ARG[key]] = value
    defaults["run_name"] = payload.get("run_name")
    return defaults


def validate_config_path(config_path: str) -> str:
    path = Path(config_path)
    if len(path.parts) != 3 or path.parts[0] != "experiments" or path.suffix not in {".yaml", ".yml"}:
        raise ValueError("Config must live under experiments/<experiment_name>/<run_name>.yaml")
    return path.parts[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument("--config")
    initial_args, _ = initial.parse_known_args(argv)
    defaults = load_yaml_defaults(initial_args.config)
    parser = build_parser(defaults=defaults)
    args = parser.parse_args(argv)
    if args.config:
        inferred_experiment = validate_config_path(args.config)
        if not args.experiment_name:
            args.experiment_name = inferred_experiment
        if not args.run_name:
            raise ValueError("run_name must be present in the YAML or overridden on the CLI.")
    else:
        if not args.experiment_name or not args.run_name:
            raise ValueError("Direct CLI usage requires both --experiment-name and --run-name.")
    return args


def args_to_config(args: argparse.Namespace) -> dict:
    raw_roots = args.raw_roots or ([] if args.data_root is None else [args.data_root])
    return {
        "run_name": args.run_name,
        "seed": args.seed,
        "data": {
            "raw_root": args.data_root,
            "raw_roots": raw_roots,
            "processed_root": args.processed_root,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "dt_seconds": args.dt_seconds,
            "history_steps": args.history_steps,
            "future_steps": args.future_steps,
            "prepare_if_missing": args.prepare_if_missing,
            "align_to_last_heading": args.align_to_last_heading,
            "chart_type": args.chart_type,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
        },
        "model": {
            "name": args.model_name,
            "num_routed_blocks": args.num_routed_blocks,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "scale_bound": args.scale_bound,
        },
        "loss": {
            "lambda_velocity": args.lambda_velocity,
            "lambda_acceleration": args.lambda_acceleration,
            "lambda_turn": args.lambda_turn,
            "lambda_feasibility": args.lambda_feasibility,
            "lambda_reverse": args.lambda_reverse,
            "lambda_position_mse": args.lambda_position_mse,
            "lambda_fde": args.lambda_fde,
            "feasibility_quantile": args.feasibility_quantile,
        },
        "optimizer": {
            "name": args.optimizer_name,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "name": args.scheduler_name,
            "factor": args.scheduler_factor,
            "patience": args.scheduler_patience,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "checkpoint_every_epochs": args.checkpoint_every_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "device": args.device,
        },
        "evaluation": {
            "save_prediction_examples": args.save_prediction_examples,
            "evaluate_reverse_mapping": args.evaluate_reverse_mapping,
        },
        "output": {
            "results_root": args.results_root,
        },
    }


def config_summary(args: argparse.Namespace) -> str:
    return json.dumps(args_to_config(args), indent=2)
