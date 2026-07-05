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
    parser.add_argument("--raw-root", default=defaults.get("raw_root"))
    parser.add_argument("--raw-roots", nargs="*", default=defaults.get("raw_roots"))
    parser.add_argument("--processed-root", default=defaults.get("processed_root"))
    parser.add_argument("--include-ship-classes", nargs="*", default=defaults.get("include_ship_classes"))
    parser.add_argument("--quality-tiers", nargs="*", default=defaults.get("quality_tiers"))
    parser.add_argument("--max-train-samples", type=int, default=defaults.get("max_train_samples"))
    parser.add_argument("--max-val-samples", type=int, default=defaults.get("max_val_samples"))
    parser.add_argument("--max-test-samples", type=int, default=defaults.get("max_test_samples"))
    parser.add_argument("--model-name", choices=["gru", "bigru", "lstm", "bilstm", "seq2seq"], default=defaults.get("model_name", "gru"))
    parser.add_argument("--hidden-dim", type=int, default=defaults.get("hidden_dim", 128))
    parser.add_argument("--num-layers", type=int, default=defaults.get("num_layers", 2))
    parser.add_argument("--loss-name", choices=["ade", "mse"], default=defaults.get("loss_name", "mse"))
    parser.add_argument("--normalize", type=_str2bool, default=defaults.get("normalize", True))
    parser.add_argument("--teacher-forcing-ratio", type=float, default=defaults.get("teacher_forcing_ratio", 0.5))
    parser.add_argument("--learning-rate", type=float, default=defaults.get("learning_rate", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 1e-4))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 80))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 256))
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 4))
    parser.add_argument("--device", default=defaults.get("device", "cuda"))
    parser.add_argument("--results-root", default=defaults.get("results_root", "baseline_results"))
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
    raw_roots = args.raw_roots or ([] if args.raw_root is None else [args.raw_root])
    return {
        "run_name": args.run_name,
        "seed": args.seed,
        "data": {
            "raw_root": args.raw_root,
            "raw_roots": raw_roots,
            "processed_root": args.processed_root,
            "include_ship_classes": args.include_ship_classes,
            "quality_tiers": args.quality_tiers,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
        },
        "model": {
            "name": args.model_name,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
        },
        "loss": {
            "name": args.loss_name,
            "normalize": args.normalize,
            "teacher_forcing_ratio": args.teacher_forcing_ratio,
        },
        "optimizer": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
        },
        "output": {
            "results_root": args.results_root,
        },
    }


