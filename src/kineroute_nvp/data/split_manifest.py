from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")
MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_ids_fingerprint(sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def processed_arrays_fingerprint(
    sample_ids: Sequence[str] | np.ndarray,
    history: np.ndarray,
    future: np.ndarray,
    history_chart: np.ndarray,
    future_chart: np.ndarray,
) -> str:
    """Hash the semantic contents of one processed split, independent of NPZ encoding."""
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    for name, values in (
        ("history", history),
        ("future", future),
        ("history_chart", history_chart),
        ("future_chart", future_chart),
    ):
        array = np.ascontiguousarray(values)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _split_file(root: Path, split: str) -> Path:
    csv_path = root / split / "part-000.csv"
    csv_gz_path = root / split / "part-000.csv.gz"
    if csv_path.exists():
        return csv_path
    if csv_gz_path.exists():
        return csv_gz_path
    raise FileNotFoundError(f"Missing {split} shard under {root}")


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def create_split_manifest(
    *,
    raw_roots: Sequence[Path],
    repo_root: Path,
    dataset_id: str,
    history_steps: int = 30,
    future_steps: int = 30,
    dt_seconds: float = 20.0,
    coordinate_unit: str,
    coordinate_frame: str,
    coordinate_unit_evidence: dict[str, str],
    sampling_interval_evidence: dict[str, str],
    training_seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    if history_steps != 30 or future_steps != 30:
        raise ValueError("The compact protocol must use exactly 30 history and 30 future points.")
    if len(set(training_seeds)) != 5:
        raise ValueError("The compact protocol must define five distinct training seeds.")

    sources: list[dict[str, Any]] = []
    split_ids: dict[str, list[str]] = {split: [] for split in SPLITS}
    for raw_root in raw_roots:
        root = Path(raw_root)
        source: dict[str, Any] = {
            "name": root.name,
            "path": _repo_relative(root, repo_root),
            "files": {},
        }
        summary_path = root / "summary.json"
        if summary_path.exists():
            source["summary_sha256"] = sha256_file(summary_path)
        for split in SPLITS:
            path = _split_file(root, split)
            coordinate_columns = ["hist_x_json", "hist_y_json", "fut_x_json", "fut_y_json"]
            frame = pd.read_csv(path, usecols=["sample_id", *coordinate_columns])
            expected_lengths = {"hist_x_json": history_steps, "hist_y_json": history_steps, "fut_x_json": future_steps, "fut_y_json": future_steps}
            for column in coordinate_columns:
                invalid = frame[column].map(lambda value: len(json.loads(value)) != expected_lengths[column])
                if bool(invalid.any()):
                    raise ValueError(f"{root.name}/{split} contains samples violating {column} length {expected_lengths[column]}.")
            ids = frame["sample_id"].astype(str).tolist()
            split_ids[split].extend(f"{root.name}::{sample_id}" for sample_id in ids)
            source["files"][split] = {
                "path": _repo_relative(path, repo_root),
                "sha256": sha256_file(path),
                "sample_count": len(ids),
                "sample_ids_sha256": sample_ids_fingerprint(ids),
                "protocol_shapes_verified": True,
            }
        sources.append(source)

    split_sets = {split: set(ids) for split, ids in split_ids.items()}
    for split, ids in split_ids.items():
        if len(ids) != len(split_sets[split]):
            raise ValueError(f"Duplicate sample IDs inside {split} split.")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(f"Split overlap between {left} and {right}: {sorted(overlap)[:3]}")

    source_fingerprint_material = []
    for source in sources:
        for split in SPLITS:
            source_fingerprint_material.append(source["files"][split]["sha256"])

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_sha256": sample_ids_fingerprint(source_fingerprint_material),
        "policy": "one frozen compact train/val/test split shared by all models and all five training seeds",
        "training_seeds": [int(seed) for seed in training_seeds],
        "sources": sources,
        "protocol": {
            "history_steps": history_steps,
            "future_steps": future_steps,
            "dt_seconds": float(dt_seconds),
            "coordinate_unit": coordinate_unit,
            "coordinate_frame": coordinate_frame,
            "coordinate_unit_evidence": coordinate_unit_evidence,
            "sampling_interval_evidence": sampling_interval_evidence,
            "split_membership_stage": "before model-specific preprocessing and normalization",
            "model_preprocessing_modes": ["raw", "heading_aligned", "heading_aligned_displacement"],
        },
        "split_counts": {split: len(split_ids[split]) for split in SPLITS},
        "split_fingerprints": {split: sample_ids_fingerprint(split_ids[split]) for split in SPLITS},
        "splits": split_ids,
    }


def _evidence_root(manifest_path: Path) -> Path:
    resolved = Path(manifest_path).resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return resolved.parent


def _read_verified_evidence(evidence: dict[str, str], evidence_root: Path) -> str:
    path = Path(evidence["path"])
    if not path.is_absolute():
        path = evidence_root / path
    if not path.exists():
        raise ValueError(f"Manifest evidence file does not exist: {path}")
    if sha256_file(path) != evidence["sha256"]:
        raise ValueError(f"Manifest evidence fingerprint mismatch: {path}")
    return path.read_text()


def validate_manifest_evidence(manifest: dict[str, Any], evidence_root: Path) -> dict[str, Any]:
    """Parse the cited source files and prove that their contents support the manifest facts."""
    protocol = manifest["protocol"]
    coordinate_text = " ".join(
        _read_verified_evidence(protocol["coordinate_unit_evidence"], evidence_root).split()
    )
    coordinate_match = re.search(
        r"All positions are in a (?P<frame>local planar frame) centred on the anchor "
        r"\(last history point\), with x\s*=\s*(?P<x>east), y\s*=\s*(?P<y>north), units of (?P<unit>metres)\.",
        coordinate_text,
        flags=re.IGNORECASE,
    )
    if coordinate_match is None:
        raise ValueError("DATA_CARD.md does not contain the coordinate-frame/unit claim recorded by the manifest.")
    parsed_unit = coordinate_match.group("unit").lower()
    parsed_frame = (
        f"{coordinate_match.group('frame').lower().removesuffix(' frame')} anchor-relative frame; "
        f"x={coordinate_match.group('x').lower()}, y={coordinate_match.group('y').lower()}"
    )
    if protocol["coordinate_unit"].lower() != parsed_unit or protocol["coordinate_frame"].lower() != parsed_frame:
        raise ValueError("Manifest coordinate facts do not match the parsed DATA_CARD.md contents.")
    coordinate_statement = " ".join(protocol["coordinate_unit_evidence"]["statement"].split()).lower()
    coordinate_statement = re.sub(r"\s*=\s*", "=", coordinate_statement)
    required_coordinate_claims = ("local planar frame", "last history point", "x=east", "y=north", parsed_unit)
    if not all(claim in coordinate_statement for claim in required_coordinate_claims):
        raise ValueError("Manifest coordinate evidence statement does not match the facts parsed from DATA_CARD.md.")

    sampling_text = _read_verified_evidence(protocol["sampling_interval_evidence"], evidence_root)

    def _markdown_integer(label: str, unit: str) -> int:
        match = re.search(rf"^\s*-\s*{re.escape(label)}:\s*(\d+)\s+{re.escape(unit)}\s*$", sampling_text, flags=re.IGNORECASE | re.MULTILINE)
        if match is None:
            raise ValueError(f"README.md does not contain a parseable '{label}' protocol statement.")
        return int(match.group(1))

    parsed_dt = _markdown_integer("Sampling interval", "seconds")
    parsed_history = _markdown_integer("History length", "points")
    parsed_future = _markdown_integer("Future length", "points")
    if (
        float(parsed_dt) != float(protocol["dt_seconds"])
        or parsed_history != int(protocol["history_steps"])
        or parsed_future != int(protocol["future_steps"])
    ):
        raise ValueError("Manifest sampling protocol does not match the parsed README.md contents.")
    sampling_statement = protocol["sampling_interval_evidence"]["statement"]
    statement_values = {}
    for label, unit in (("Sampling interval", "seconds"), ("History length", "points"), ("Future length", "points")):
        match = re.search(rf"{re.escape(label)}:\s*(\d+)\s+{re.escape(unit)}", sampling_statement, flags=re.IGNORECASE)
        if match is None:
            raise ValueError("Manifest sampling evidence statement is not parseable.")
        statement_values[label] = int(match.group(1))
    if statement_values != {
        "Sampling interval": parsed_dt,
        "History length": parsed_history,
        "Future length": parsed_future,
    }:
        raise ValueError("Manifest sampling evidence statement does not match the facts parsed from README.md.")
    return {
        "coordinate_excerpt": coordinate_match.group(0),
        "coordinate_unit": parsed_unit,
        "coordinate_frame": parsed_frame,
        "dt_seconds": float(parsed_dt),
        "history_steps": parsed_history,
        "future_steps": parsed_future,
    }


def load_split_manifest(path: Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest_integrity(manifest)
    validate_manifest_evidence(manifest, _evidence_root(manifest_path))
    return manifest


def validate_manifest_integrity(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported split manifest schema: {manifest.get('schema_version')}")
    seeds = manifest.get("training_seeds", [])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Split manifest must define five distinct training seeds.")
    protocol = manifest.get("protocol", {})
    if protocol.get("history_steps") != 30 or protocol.get("future_steps") != 30:
        raise ValueError("Split manifest does not describe the required 30 -> 30 protocol.")
    if float(protocol.get("dt_seconds", -1.0)) != 20.0:
        raise ValueError("Split manifest does not preserve the expected 20-second sampling interval.")
    if not protocol.get("sampling_interval_evidence"):
        raise ValueError("Split manifest must record sampling-interval evidence.")

    source_hashes = [source["files"][split]["sha256"] for source in manifest.get("sources", []) for split in SPLITS]
    if sample_ids_fingerprint(source_hashes) != manifest.get("dataset_sha256"):
        raise ValueError("Manifest dataset fingerprint does not match its frozen source fingerprints.")
    unit_evidence = protocol.get("coordinate_unit_evidence", {})
    if not protocol.get("coordinate_unit") or not unit_evidence.get("path") or not unit_evidence.get("sha256") or not unit_evidence.get("statement"):
        raise ValueError("Split manifest must document coordinate units and their evidence.")

    split_sets: dict[str, set[str]] = {}
    for split in SPLITS:
        ids = [str(value) for value in manifest.get("splits", {}).get(split, [])]
        expected_count = int(manifest.get("split_counts", {}).get(split, -1))
        expected_hash = manifest.get("split_fingerprints", {}).get(split)
        if len(ids) != expected_count:
            raise ValueError(f"Manifest {split} count mismatch: {len(ids)} != {expected_count}")
        if sample_ids_fingerprint(ids) != expected_hash:
            raise ValueError(f"Manifest {split} fingerprint mismatch.")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Manifest {split} contains duplicate sample IDs.")
        source_count = sum(int(source["files"][split]["sample_count"]) for source in manifest["sources"])
        if source_count != expected_count:
            raise ValueError(f"Manifest source counts for {split} do not sum to the frozen split count.")
        split_sets[split] = set(ids)
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(f"Manifest split overlap between {left} and {right}.")


def validate_split_membership(split: str, sample_ids: Sequence[str], manifest: dict[str, Any]) -> None:
    actual = [str(value) for value in sample_ids]
    expected = [str(value) for value in manifest["splits"][split]]
    if actual != expected:
        actual_set = set(actual)
        expected_set = set(expected)
        missing = sorted(expected_set - actual_set)[:3]
        unexpected = sorted(actual_set - expected_set)[:3]
        raise ValueError(
            f"Frozen {split} split mismatch: expected {len(expected)} ordered samples, got {len(actual)}; "
            f"missing={missing}, unexpected={unexpected}. Refusing to repartition."
        )


def validate_raw_sources(raw_roots: Sequence[Path], manifest: dict[str, Any]) -> None:
    expected_sources = manifest.get("sources", [])
    if len(raw_roots) != len(expected_sources):
        raise ValueError(f"Frozen dataset expects {len(expected_sources)} source roots, got {len(raw_roots)}.")
    for root_value, source in zip(raw_roots, expected_sources):
        root = Path(root_value)
        if root.name != source["name"]:
            raise ValueError(f"Frozen source order/name mismatch: expected {source['name']}, got {root.name}.")
        summary_path = root / "summary.json"
        if "summary_sha256" in source:
            if not summary_path.exists() or sha256_file(summary_path) != source["summary_sha256"]:
                raise ValueError(f"Frozen summary fingerprint mismatch for {root.name}.")
        for split in SPLITS:
            actual_hash = sha256_file(_split_file(root, split))
            expected_hash = source["files"][split]["sha256"]
            if actual_hash != expected_hash:
                raise ValueError(f"Frozen source fingerprint mismatch for {root.name}/{split}; refusing changed data.")


def validate_experiment_protocol(config: dict[str, Any], metadata: dict[str, Any], manifest: dict[str, Any] | None = None) -> None:
    data = config["data"]
    expected = manifest["protocol"] if manifest is not None else {
        "history_steps": 30,
        "future_steps": 30,
        "dt_seconds": metadata.get("dt_seconds", data.get("dt_seconds", 20.0)),
    }
    for key in ("history_steps", "future_steps"):
        configured = int(data.get(key, 30))
        required = int(expected[key])
        if configured != required or required != 30:
            raise ValueError(f"Protocol violation: {key} must be 30, got configured={configured}, expected={required}.")
        if key in metadata and int(metadata[key]) != required:
            raise ValueError(f"Processed metadata {key}={metadata[key]} does not match frozen value {required}.")
    configured_dt = float(data.get("dt_seconds", 20.0))
    required_dt = float(expected["dt_seconds"])
    metadata_dt = float(metadata.get("dt_seconds", required_dt))
    if configured_dt != required_dt or metadata_dt != required_dt:
        raise ValueError(
            f"Protocol violation: dt_seconds must remain {required_dt}, got config={configured_dt}, metadata={metadata_dt}."
        )
    chart_models = {"kineroute_nvp", "routed_realnvp", "realnvp", "non_invertible_routed"}
    if config.get("model", {}).get("name") in chart_models:
        configured_chart = data.get("chart_type")
        configured_alignment = bool(data.get("align_to_last_heading", False))
        metadata_chart = metadata.get("chart_type")
        metadata_alignment = bool(metadata.get("align_to_last_heading", False))
        if metadata_chart != configured_chart or metadata_alignment != configured_alignment:
            raise ValueError(
                "Processed chart preprocessing disagrees with the resolved chart-model config: "
                f"config=(chart_type={configured_chart}, align={configured_alignment}), "
                f"metadata=(chart_type={metadata_chart}, align={metadata_alignment})."
            )
    if manifest is not None:
        for key in ("dataset_id",):
            if metadata.get(key) is not None and metadata[key] != manifest[key]:
                raise ValueError(f"Processed metadata {key} does not match the frozen manifest.")
        for key in ("coordinate_unit", "coordinate_frame"):
            expected_value = manifest["protocol"][key]
            if metadata.get(key) is not None and metadata[key] != expected_value:
                raise ValueError(f"Processed metadata {key} does not match the frozen manifest.")


def validate_processed_dataset(processed_root: Path, manifest_path: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(processed_root)
    manifest = load_split_manifest(manifest_path)
    metadata = json.loads((root / "metadata.json").read_text())
    recorded_processed_fingerprints = metadata.get("processed_split_fingerprints")
    if not isinstance(recorded_processed_fingerprints, dict):
        raise ValueError("Processed metadata lacks semantic split fingerprints; explicitly regenerate this legacy dataset.")
    if metadata.get("split_fingerprints") is not None and metadata["split_fingerprints"] != manifest["split_fingerprints"]:
        raise ValueError("Processed metadata split fingerprints do not match the frozen manifest.")
    for split in SPLITS:
        with np.load(root / f"{split}.npz", allow_pickle=True) as payload:
            sample_ids = payload["sample_id"].astype(str).tolist()
            history = payload["history"]
            future = payload["future"]
            history_chart = payload["history_chart"]
            future_chart = payload["future_chart"]
            history_shape = history.shape
            future_shape = future.shape
        validate_split_membership(split, sample_ids, manifest)
        expected_count = manifest["split_counts"][split]
        if history_shape != (expected_count, 30, 2) or future_shape != (expected_count, 30, 2):
            raise ValueError(
                f"Processed {split} violates frozen 30 -> 30 protocol: history={history_shape}, future={future_shape}."
            )
        if int(metadata.get("split_sizes", {}).get(split, -1)) != expected_count:
            raise ValueError(f"Processed metadata count for {split} does not match the frozen manifest.")
        actual_processed_fingerprint = processed_arrays_fingerprint(
            sample_ids, history, future, history_chart, future_chart
        )
        if recorded_processed_fingerprints.get(split) != actual_processed_fingerprint:
            raise ValueError(f"Processed semantic fingerprint mismatch for {split}; refusing drifted arrays.")
    if config is not None:
        validate_experiment_protocol(config, metadata, manifest)
    return manifest
