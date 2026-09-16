from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from kineroute_nvp.geometry.kinematics import (
    describe_chart_choice,
    describe_displacement_chart_choice,
    describe_position_chart_choice,
    encode_positions_numpy,
)
from kineroute_nvp.data.split_manifest import (
    load_split_manifest,
    processed_arrays_fingerprint,
    validate_raw_sources,
    validate_split_membership,
)
from kineroute_nvp.data.trajectory_preprocessing import encode_with_history_torch
from kineroute_nvp.utils.io import atomic_write_json


REQUIRED_COLUMNS = ["sample_id", "hist_x_json", "hist_y_json", "fut_x_json", "fut_y_json"]


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


@dataclass
class PrepareSummary:
    processed_root: Path
    metadata_path: Path
    normalization_path: Path
    thresholds_path: Path
    split_sizes: Dict[str, int]


def _parse_positions(row: pd.Series, prefix: str) -> np.ndarray:
    x = np.asarray(json.loads(row[f"{prefix}_x_json"]), dtype=np.float64)
    y = np.asarray(json.loads(row[f"{prefix}_y_json"]), dtype=np.float64)
    if x.shape != (30,) or y.shape != (30,):
        raise ValueError(f"{prefix} arrays must be length 30, got {x.shape} and {y.shape}")
    return np.stack([x, y], axis=-1)


def _compute_thresholds(train_future: np.ndarray, train_history: np.ndarray, dt_seconds: float, quantile: float) -> Dict[str, float]:
    last_history = train_history[:, -1:, :]
    sequence = np.concatenate([last_history, train_future], axis=1)
    velocity = np.diff(sequence, axis=1) / dt_seconds
    acceleration = np.diff(velocity, axis=1) / dt_seconds
    speed = np.linalg.norm(velocity, axis=-1)
    heading = np.zeros_like(speed)
    raw_heading = np.arctan2(velocity[..., 1], velocity[..., 0])
    heading[:, 0] = np.where(speed[:, 0] > 1e-8, raw_heading[:, 0], 0.0)
    for idx in range(1, speed.shape[1]):
        heading[:, idx] = np.where(speed[:, idx] > 1e-8, raw_heading[:, idx], heading[:, idx - 1])
    turn_rate = np.arctan2(np.sin(heading[:, 1:] - heading[:, :-1]), np.cos(heading[:, 1:] - heading[:, :-1])) / dt_seconds
    return {
        "max_acceleration": float(np.quantile(np.linalg.norm(acceleration, axis=-1), quantile)),
        "max_turn_rate": float(np.quantile(np.abs(turn_rate), quantile)),
    }


def _align_pair_to_last_heading(history: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Delegate to the same batched Torch implementation used online by
    # ordinary forecasters, so persistent chart data cannot drift numerically
    # or semantically from their preprocessing path.
    history_tensor = torch.as_tensor(history, dtype=torch.float32).unsqueeze(0)
    future_tensor = torch.as_tensor(future, dtype=torch.float32).unsqueeze(0)
    return (
        encode_with_history_torch(history_tensor, history_tensor, "heading_aligned")[0].numpy(),
        encode_with_history_torch(history_tensor, future_tensor, "heading_aligned")[0].numpy(),
    )


def _load_split_frame(raw_root: Path, split: str) -> pd.DataFrame:
    csv_path = raw_root / split / "part-000.csv"
    csv_gz_path = raw_root / split / "part-000.csv.gz"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if csv_gz_path.exists():
        return pd.read_csv(csv_gz_path)
    raise FileNotFoundError(f"Missing raw split file under {raw_root / split}: expected part-000.csv or part-000.csv.gz")


def prepare_envship_dataset(
    raw_root: Path | None,
    processed_root: Path,
    dt_seconds: float,
    feasibility_quantile: float,
    align_to_last_heading: bool = False,
    chart_type: str = "polar",
    splits: Iterable[str] = ("train", "val", "test"),
    raw_roots: Sequence[Path] | None = None,
    include_ship_classes: Sequence[str] | None = None,
    quality_tiers: Sequence[str] | None = None,
    split_manifest_path: Path | None = None,
) -> PrepareSummary:
    candidate_roots = ([Path(raw_root)] if raw_root is not None else []) + ([Path(root) for root in raw_roots] if raw_roots else [])
    resolved_raw_roots = []
    seen_roots = set()
    for root in candidate_roots:
        resolved = root.resolve()
        if resolved not in seen_roots:
            resolved_raw_roots.append(root)
            seen_roots.add(resolved)
    if not resolved_raw_roots:
        raise ValueError("At least one raw_root must be provided.")
    split_manifest = load_split_manifest(split_manifest_path) if split_manifest_path is not None else None
    if split_manifest is not None:
        validate_raw_sources(resolved_raw_roots, split_manifest)
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    split_sizes: Dict[str, int] = {}
    train_history = None
    train_future = None
    train_history_chart = None
    train_future_chart = None
    processed_split_fingerprints: Dict[str, str] = {}

    for split in splits:
        frames = []
        for root in resolved_raw_roots:
            source_frame = _load_split_frame(root, split)
            if split_manifest is not None:
                source_frame = source_frame.copy()
                source_frame["sample_id"] = root.name + "::" + source_frame["sample_id"].astype(str)
            frames.append(source_frame)
        frame = pd.concat(frames, ignore_index=True)
        if include_ship_classes:
            frame = frame[frame["ship_class"].isin(include_ship_classes)].reset_index(drop=True)
        if quality_tiers:
            frame = frame[frame["quality_tier"].isin(quality_tiers)].reset_index(drop=True)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns in split {split}: {missing}")
        if split_manifest is not None:
            validate_split_membership(split, frame["sample_id"].astype(str).tolist(), split_manifest)
        histories = []
        futures = []
        sample_ids = []
        for _, row in frame[REQUIRED_COLUMNS].iterrows():
            history = _parse_positions(row, "hist")
            future = _parse_positions(row, "fut")
            histories.append(history)
            futures.append(future)
            sample_ids.append(row["sample_id"])
        history_array = np.stack(histories).astype(np.float32)
        future_array = np.stack(futures).astype(np.float32)
        if align_to_last_heading:
            history_tensor = torch.from_numpy(history_array)
            future_tensor = torch.from_numpy(future_array)
            history_array = encode_with_history_torch(
                history_tensor, history_tensor, "heading_aligned"
            ).numpy()
            future_array = encode_with_history_torch(
                history_tensor, future_tensor, "heading_aligned"
            ).numpy()
        history_chart = encode_positions_numpy(history_array, chart_type=chart_type).astype(np.float32)
        future_chart = encode_positions_numpy(future_array, chart_type=chart_type).astype(np.float32)
        _atomic_savez(
            processed_root / f"{split}.npz",
            sample_id=np.asarray(sample_ids),
            history=history_array,
            future=future_array,
            history_chart=history_chart,
            future_chart=future_chart,
        )
        processed_split_fingerprints[split] = processed_arrays_fingerprint(
            sample_ids, history_array, future_array, history_chart, future_chart
        )
        split_sizes[split] = int(history_array.shape[0])
        if split == "train":
            train_history = history_array
            train_future = future_array
            train_history_chart = history_chart
            train_future_chart = future_chart

    assert train_history is not None and train_future is not None
    assert train_history_chart is not None and train_future_chart is not None
    history_chart_mean = train_history_chart.mean(axis=0)
    history_chart_std = np.where(train_history_chart.std(axis=0) < 1e-6, 1.0, train_history_chart.std(axis=0))
    future_chart_mean = train_future_chart.mean(axis=0)
    future_chart_std = np.where(train_future_chart.std(axis=0) < 1e-6, 1.0, train_future_chart.std(axis=0))
    normalization = {
        "history_chart_mean": history_chart_mean.tolist(),
        "history_chart_std": history_chart_std.tolist(),
        "future_chart_mean": future_chart_mean.tolist(),
        "future_chart_std": future_chart_std.tolist(),
    }
    thresholds = _compute_thresholds(train_future, train_history, dt_seconds=dt_seconds, quantile=feasibility_quantile)

    normalization_path = processed_root / "normalization.json"
    thresholds_path = processed_root / "feasibility_thresholds.json"
    metadata_path = processed_root / "metadata.json"
    atomic_write_json(normalization_path, normalization)
    atomic_write_json(thresholds_path, thresholds)
    metadata = {
        "raw_roots": [str(root) for root in resolved_raw_roots],
        "processed_root": str(processed_root),
        "dt_seconds": dt_seconds,
        "history_steps": 30,
        "future_steps": 30,
        "coordinate_unit": split_manifest["protocol"]["coordinate_unit"] if split_manifest is not None else "unspecified",
        "coordinate_frame": split_manifest["protocol"]["coordinate_frame"] if split_manifest is not None else "unspecified",
        "split_manifest": str(split_manifest_path) if split_manifest_path is not None else None,
        "split_fingerprints": split_manifest["split_fingerprints"] if split_manifest is not None else None,
        "dataset_id": split_manifest["dataset_id"] if split_manifest is not None else None,
        "chart": (
            describe_chart_choice()
            if chart_type == "polar"
            else describe_displacement_chart_choice()
            if chart_type == "displacement"
            else describe_position_chart_choice()
        ),
        "chart_type": chart_type,
        "split_sizes": split_sizes,
        "processed_split_fingerprints": processed_split_fingerprints,
        "required_columns": REQUIRED_COLUMNS,
        "align_to_last_heading": align_to_last_heading,
        "include_ship_classes": list(include_ship_classes) if include_ship_classes else None,
        "quality_tiers": list(quality_tiers) if quality_tiers else None,
        "normalization_path": str(normalization_path),
        "thresholds_path": str(thresholds_path),
        "feasibility_quantile": feasibility_quantile,
    }
    atomic_write_json(metadata_path, metadata)
    return PrepareSummary(
        processed_root=processed_root,
        metadata_path=metadata_path,
        normalization_path=normalization_path,
        thresholds_path=thresholds_path,
        split_sizes=split_sizes,
    )
