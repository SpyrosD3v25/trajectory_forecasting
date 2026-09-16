from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate result folders into one CSV row per run.")
    parser.add_argument("result_folders", nargs="*", help="Explicit result folders. Defaults to every _results.json under --results-root.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/aggregate.csv")
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _result_paths(args: argparse.Namespace) -> list[Path]:
    if args.result_folders:
        return [(_resolve(folder) / "_results.json") if _resolve(folder).is_dir() else _resolve(folder) for folder in args.result_folders]
    return sorted(_resolve(args.results_root).glob("*/*_results.json")) or sorted(_resolve(args.results_root).glob("*/_results.json"))


def _get(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def row_from_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("resolved_configuration", {})
    model = config.get("model", {})
    data = config.get("data", {})
    loss = config.get("loss", {})
    training = config.get("training", {})
    return {
        "result_dir": str(path.parent),
        "status": payload.get("status"),
        "artifact_schema_version": payload.get("artifact_schema_version"),
        "experiment_name": payload.get("experiment_name"),
        "run_name": payload.get("run_name"),
        "seed": payload.get("seed", config.get("seed")),
        "model": payload.get("model", model.get("name")),
        "preprocessing_mode": data.get("preprocessing_mode"),
        "loss": loss.get("name"),
        "hidden_dim": model.get("hidden_dim"),
        "num_layers": model.get("num_layers"),
        "num_blocks": model.get("num_blocks", model.get("num_routed_blocks")),
        "epochs": training.get("epochs"),
        "batch_size": training.get("batch_size"),
        "params": payload.get("model_parameter_count"),
        "best_epoch": payload.get("best_epoch"),
        "best_validation_ade": payload.get("best_validation_ade"),
        "best_validation_fde": payload.get("best_validation_fde"),
        "test_ade": payload.get("test_ade"),
        "test_fde": payload.get("test_fde"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "manifest_sha256": _get(payload, "dataset_identity", "manifest_sha256"),
        "runtime_source_sha256": _get(payload, "runtime_source_identity", "sha256"),
        "checkpoint_best": _get(payload, "checkpoint_paths", "best"),
        "checkpoint_final": _get(payload, "checkpoint_paths", "final"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = []
    for path in _result_paths(args):
        if path.is_file():
            rows.append(row_from_payload(path, json.loads(path.read_text())))
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "result_dir",
        "status",
        "artifact_schema_version",
        "experiment_name",
        "run_name",
        "seed",
        "model",
        "preprocessing_mode",
        "loss",
        "hidden_dim",
        "num_layers",
        "num_blocks",
        "epochs",
        "batch_size",
        "params",
        "best_epoch",
        "best_validation_ade",
        "best_validation_fde",
        "test_ade",
        "test_fde",
        "runtime_seconds",
        "manifest_sha256",
        "runtime_source_sha256",
        "checkpoint_best",
        "checkpoint_final",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
