from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("best_validation_ade", "best_validation_fde", "test_ade", "test_fde")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly aggregate identical experiment definitions across training seeds.")
    parser.add_argument("result_folders", nargs="*")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/multiseed_aggregate.csv")
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--allow-partial", action="store_true", help="Allow a subset of expected seeds for debugging.")
    parser.add_argument(
        "--allow-unfrozen-data",
        action="store_true",
        help="Allow artifacts without dataset_identity for smoke/debug use only.",
    )
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _result_paths(args: argparse.Namespace) -> list[Path]:
    if args.result_folders:
        result = []
        for value in args.result_folders:
            path = _resolve(value)
            result.append(path / "_results.json" if path.is_dir() else path)
        return result
    root = _resolve(args.results_root)
    return sorted(root.glob("*/_results.json"))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _definition(payload: dict[str, Any]) -> tuple[str, str]:
    config = copy.deepcopy(payload["resolved_configuration"])
    config.pop("seed", None)
    config.pop("run_name", None)
    config.get("output", {}).pop("results_root", None)
    definition = {
        "experiment_name": payload.get("experiment_name"),
        "config_except_seed": config,
        "dataset_identity": payload.get("dataset_identity"),
        "runtime_source_identity": payload.get("runtime_source_identity"),
    }
    canonical = _canonical(definition)
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metric(payload: dict[str, Any], name: str, path: Path) -> float:
    try:
        value = float(payload[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: missing or invalid {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: {name} must be finite")
    return value


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expected_seeds = set(args.expected_seeds)
    if len(expected_seeds) != len(args.expected_seeds):
        raise ValueError("--expected-seeds contains duplicates")
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    identities: dict[str, str] = {}
    for path in _result_paths(args):
        if not path.is_file():
            raise FileNotFoundError(f"Missing result artifact: {path}")
        payload = json.loads(path.read_text())
        if payload.get("status") != "completed":
            raise ValueError(f"{path}: multi-seed aggregation requires status=completed")
        if payload.get("artifact_schema_version") != 2:
            raise ValueError(f"{path}: multi-seed aggregation requires artifact schema version 2")
        if not isinstance(payload.get("resolved_configuration"), dict):
            raise ValueError(f"{path}: missing resolved_configuration")
        if payload.get("dataset_identity") is None and not args.allow_unfrozen_data:
            raise ValueError(f"{path}: missing dataset_identity; use --allow-unfrozen-data only for smoke/debug data")
        if payload.get("runtime_source_identity") is None:
            raise ValueError(f"{path}: missing runtime_source_identity")
        seed = payload.get("seed")
        if seed != payload["resolved_configuration"].get("seed"):
            raise ValueError(f"{path}: artifact seed disagrees with resolved configuration")
        if seed not in expected_seeds:
            raise ValueError(f"{path}: unexpected seed {seed}; expected {sorted(expected_seeds)}")
        for metric in METRICS:
            _metric(payload, metric, path)
        definition, identity_hash = _definition(payload)
        identities[definition] = identity_hash
        groups[definition].append((path, payload))

    if not groups:
        raise ValueError("No result artifacts found")

    output_rows: list[dict[str, Any]] = []
    for definition, items in groups.items():
        by_seed: dict[int, tuple[Path, dict[str, Any]]] = {}
        for path, payload in items:
            seed = int(payload["seed"])
            if seed in by_seed:
                raise ValueError(f"Duplicate seed {seed} for experiment definition {identities[definition]}")
            by_seed[seed] = (path, payload)
        seeds = set(by_seed)
        if not args.allow_partial and seeds != expected_seeds:
            raise ValueError(
                f"Incomplete seed set for experiment definition {identities[definition]}: "
                f"expected {sorted(expected_seeds)}, got {sorted(seeds)}"
            )
        first = next(iter(by_seed.values()))[1]
        config = first["resolved_configuration"]
        values = {metric: [_metric(by_seed[seed][1], metric, by_seed[seed][0]) for seed in sorted(seeds)] for metric in METRICS}
        output_rows.append(
            {
                "experiment_name": first.get("experiment_name"),
                "model": config.get("model", {}).get("name"),
                "preprocessing_mode": config.get("data", {}).get("preprocessing_mode"),
                "loss": config.get("loss", {}).get("name"),
                "seeds": " ".join(str(seed) for seed in sorted(seeds)),
                "runs": len(seeds),
                "config_identity": identities[definition],
                "manifest_sha256": (first.get("dataset_identity") or {}).get("manifest_sha256", "UNFROZEN_DEBUG"),
                "runtime_source_sha256": (first.get("runtime_source_identity") or {}).get("sha256"),
                "val_ade_mean": statistics.mean(values["best_validation_ade"]),
                "val_ade_std": _std(values["best_validation_ade"]),
                "val_fde_mean": statistics.mean(values["best_validation_fde"]),
                "val_fde_std": _std(values["best_validation_fde"]),
                "test_ade_mean": statistics.mean(values["test_ade"]),
                "test_ade_std": _std(values["test_ade"]),
                "test_fde_mean": statistics.mean(values["test_fde"]),
                "test_fde_std": _std(values["test_fde"]),
            }
        )

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_name",
        "model",
        "preprocessing_mode",
        "loss",
        "seeds",
        "runs",
        "config_identity",
        "manifest_sha256",
        "runtime_source_sha256",
        "val_ade_mean",
        "val_ade_std",
        "val_fde_mean",
        "val_fde_std",
        "test_ade_mean",
        "test_ade_std",
        "test_fde_mean",
        "test_fde_std",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(output_rows, key=lambda row: (str(row["experiment_name"]), str(row["model"]), str(row["config_identity"]))))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
