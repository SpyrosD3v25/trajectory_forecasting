from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CELLS = {
    ("gru", "raw", "mse"),
    ("gru", "heading_aligned", "mse"),
    ("gru", "heading_aligned_displacement", "mse"),
    ("gru", "heading_aligned_displacement", "ade"),
    ("lstm", "raw", "mse"),
    ("lstm", "heading_aligned", "mse"),
    ("lstm", "heading_aligned_displacement", "mse"),
    ("lstm", "heading_aligned_displacement", "ade"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and create the Section 4 preprocessing-attribution table.")
    parser.add_argument("result_folders", nargs="*", help="Explicit result folders; otherwise scan --results-root.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/preprocessing_ablation.csv")
    parser.add_argument(
        "--selection-output",
        help="Selection JSON path (defaults beside --output); emitted only for a complete matrix.",
    )
    parser.add_argument("--expected-seed", type=int, default=0)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit a compatible subset for debugging. Final mode requires all eight cells.",
    )
    return parser.parse_args(argv)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _result_paths(args: argparse.Namespace) -> list[Path]:
    if args.result_folders:
        paths = []
        for folder in args.result_folders:
            path = _resolve(folder)
            paths.append(path / "_results.json" if path.is_dir() else path)
        return paths
    return sorted(_resolve(args.results_root).glob("*/_results.json"))


def _require_finite(payload: dict[str, Any], key: str, path: Path) -> float:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: missing or invalid {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: {key} must be finite, got {value}")
    return value


def _without_intentional_factors(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only the factors intentionally varied by the Section 4 matrix."""
    controlled = copy.deepcopy(config)
    controlled.pop("run_name", None)
    controlled.pop("seed", None)
    for section, keys in {
        "data": ("preprocessing_mode", "align_to_last_heading", "chart_type"),
        "model": ("name",),
        "loss": ("name",),
    }.items():
        values = controlled.get(section, {})
        for key in keys:
            values.pop(key, None)
    return controlled


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validated_row(
    path: Path,
    payload: dict[str, Any],
    expected_seed: int,
) -> tuple[dict[str, Any], tuple[str, str, str], str, str, str]:
    if payload.get("status") != "completed":
        raise ValueError(f"{path}: Section 4 aggregation requires status=completed")
    if payload.get("artifact_schema_version") != 2:
        raise ValueError(f"{path}: official Section 4 aggregation requires artifact schema version 2")
    config = payload.get("resolved_configuration")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: missing resolved_configuration")
    seed = payload.get("seed", config.get("seed"))
    if seed != config.get("seed") or seed != expected_seed:
        raise ValueError(f"{path}: expected the single attribution seed {expected_seed}, got {seed}")
    data = config.get("data", {})
    model_config = config.get("model", {})
    loss = config.get("loss", {})
    model = payload.get("model", model_config.get("name"))
    if model != model_config.get("name"):
        raise ValueError(f"{path}: result model disagrees with resolved configuration")
    mode = data.get("preprocessing_mode")
    loss_name = loss.get("name")
    cell = (model, mode, loss_name)
    if cell not in EXPECTED_CELLS:
        raise ValueError(f"{path}: unexpected preprocessing-ablation cell {cell}")

    identity = payload.get("dataset_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{path}: missing immutable dataset_identity provenance")
    required_identity = {
        "manifest_path",
        "manifest_sha256",
        "dataset_id",
        "dataset_sha256",
        "split_counts",
        "split_fingerprints",
        "processed_split_fingerprints",
        "processed_chart_type",
        "processed_align_to_last_heading",
        "protocol",
    }
    missing_identity = sorted(required_identity - identity.keys())
    if missing_identity:
        raise ValueError(f"{path}: incomplete dataset_identity: {missing_identity}")
    if payload.get("dataset_split_sizes") != identity["split_counts"]:
        raise ValueError(f"{path}: result split counts disagree with dataset identity")
    source_identity = payload.get("runtime_source_identity")
    if not isinstance(source_identity, dict) or not source_identity.get("sha256"):
        raise ValueError(f"{path}: missing runtime_source_identity provenance")

    params = _require_finite(payload, "model_parameter_count", path)
    if params <= 0 or not params.is_integer():
        raise ValueError(f"{path}: model_parameter_count must be a positive integer")
    if payload.get("test_evaluated") is not False:
        raise ValueError(f"{path}: official Section 4 selection runs must be validation-only")
    if payload.get("test_ade") is not None or payload.get("test_fde") is not None:
        raise ValueError(f"{path}: validation-only run unexpectedly contains test metrics")
    metrics = {
        key: _require_finite(payload, key, path)
        for key in ("best_validation_ade", "best_validation_fde")
    }
    row = {
        "run_name": payload.get("run_name"),
        "seed": seed,
        "model": model,
        "alignment": mode in {"heading_aligned", "heading_aligned_displacement"},
        "displacement": mode == "heading_aligned_displacement",
        "preprocessing_mode": mode,
        "loss": loss_name,
        "params": int(params),
        **metrics,
        "test_ade": None,
        "test_fde": None,
        "manifest_sha256": identity["manifest_sha256"],
        "runtime_source_sha256": source_identity["sha256"],
        "result_dir": str(path.parent),
    }
    return row, cell, _canonical(_without_intentional_factors(config)), _canonical(identity), _canonical(source_identity)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, Any]] = []
    cells: set[tuple[str, str, str]] = set()
    control_signatures: set[str] = set()
    dataset_identities: set[str] = set()
    source_identities: set[str] = set()
    for path in _result_paths(args):
        if not path.is_file():
            raise FileNotFoundError(f"Missing result artifact: {path}")
        payload = json.loads(path.read_text())
        if payload.get("experiment_name") != "preprocessing_ablation":
            continue
        row, cell, control_signature, dataset_identity, source_identity = _validated_row(path, payload, args.expected_seed)
        if cell in cells:
            raise ValueError(f"Duplicate preprocessing-ablation cell: {cell}")
        cells.add(cell)
        rows.append(row)
        control_signatures.add(control_signature)
        dataset_identities.add(dataset_identity)
        source_identities.add(source_identity)

    if not rows:
        raise ValueError("No completed preprocessing_ablation results found")
    if len(control_signatures) != 1:
        raise ValueError("Runs differ in controlled configuration fields")
    if len(dataset_identities) != 1:
        raise ValueError("Runs use different dataset/manifest identities")
    if len(source_identities) != 1:
        raise ValueError("Runs use different runtime source snapshots")
    if not args.allow_partial and cells != EXPECTED_CELLS:
        missing = sorted(EXPECTED_CELLS - cells)
        raise ValueError(f"Incomplete preprocessing-ablation matrix; missing cells: {missing}")

    rows.sort(key=lambda row: (str(row["model"]), str(row["preprocessing_mode"]), str(row["loss"])))
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "seed",
        "model",
        "alignment",
        "displacement",
        "preprocessing_mode",
        "loss",
        "params",
        "best_validation_ade",
        "best_validation_fde",
        "test_ade",
        "test_fde",
        "manifest_sha256",
        "runtime_source_sha256",
        "result_dir",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if not args.allow_partial:
        protocol_order = [
            ("raw", "mse"),
            ("heading_aligned", "mse"),
            ("heading_aligned_displacement", "mse"),
            ("heading_aligned_displacement", "ade"),
        ]
        summaries = []
        for order, (mode, loss_name) in enumerate(protocol_order):
            protocol_rows = [
                row
                for row in rows
                if row["preprocessing_mode"] == mode and row["loss"] == loss_name
            ]
            if {row["model"] for row in protocol_rows} != {"gru", "lstm"}:
                raise ValueError(f"Protocol {(mode, loss_name)} does not contain exactly GRU and LSTM")
            summaries.append(
                {
                    "preprocessing_mode": mode,
                    "loss": loss_name,
                    "mean_validation_ade": sum(row["best_validation_ade"] for row in protocol_rows) / 2.0,
                    "mean_validation_fde": sum(row["best_validation_fde"] for row in protocol_rows) / 2.0,
                    "tie_break_order": order,
                }
            )
        selected = min(
            summaries,
            key=lambda item: (
                item["mean_validation_ade"],
                item["mean_validation_fde"],
                item["tie_break_order"],
            ),
        )
        selection_path = (
            _resolve(args.selection_output)
            if args.selection_output
            else output.with_name(f"{output.stem}_selection.json")
        )
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(
            json.dumps(
                {
                    "selection_metric": "lowest mean validation ADE across GRU and LSTM",
                    "tie_break": "mean validation FDE, then predeclared simpler-protocol order",
                    "test_metrics_used_for_selection": False,
                    "selected": selected,
                    "protocols": summaries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(selection_path)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
