from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import kineroute_nvp.training.trainer as trainer_module
from kineroute_nvp.data.dataset import ProcessedTrajectoryDataset
from kineroute_nvp.data.preprocess import _align_pair_to_last_heading, prepare_envship_dataset
from kineroute_nvp.data.split_manifest import (
    SPLITS,
    create_split_manifest,
    load_split_manifest,
    sample_ids_fingerprint,
    sha256_file,
    validate_experiment_protocol,
    validate_manifest_evidence,
    validate_raw_sources,
    validate_processed_dataset,
    validate_split_membership,
)
from kineroute_nvp.data.trajectory_preprocessing import (
    decode_with_history_torch,
    encode_history_future,
    encode_trajectory,
    encode_with_history_torch,
)
from kineroute_nvp.geometry.kinematics import encode_positions_numpy
from kineroute_nvp.models.factory import build_model
from kineroute_nvp.training.trainer import CHART_MODELS, ExperimentRunner
from kineroute_nvp.utils.config import args_to_config, parse_args
from kineroute_nvp.utils.model import count_trainable_parameters


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPACT_MANIFEST = REPO_ROOT / "experiments/compact/split_manifest.json"
COMPACT_RAW_ROOTS = [
    REPO_ROOT / "data/envship/paper_dma_clean_ship_core_lite_v1",
    REPO_ROOT / "data/envship/paper_noaa_clean_ship_core_lite_v1",
]


def test_persistent_heading_alignment_anchors_at_final_history_point() -> None:
    history = _trajectory(offset=137.0)
    delta = history[-1] - history[-2]
    future = history[-1:] + np.arange(1, 31, dtype=np.float64)[:, None] * delta

    aligned_history, aligned_future = _align_pair_to_last_heading(history, future)

    np.testing.assert_allclose(aligned_history[-1], [0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(aligned_future[:, 1], 0.0, atol=3e-6)
    assert np.all(aligned_future[:, 0] > 0.0)


def test_persistent_chart_and_online_displacement_preprocessing_are_identical() -> None:
    histories = np.stack([_trajectory(offset=137.0), _trajectory(offset=-51.0, curve=-0.01)]).astype(np.float32)
    futures = histories + np.asarray([30.0, -17.0], dtype=np.float32)
    history_tensor = torch.from_numpy(histories)
    future_tensor = torch.from_numpy(futures)
    online_history = encode_with_history_torch(history_tensor, history_tensor, "heading_aligned_displacement").numpy()
    online_future = encode_with_history_torch(history_tensor, future_tensor, "heading_aligned_displacement").numpy()
    aligned_history = encode_with_history_torch(history_tensor, history_tensor, "heading_aligned").numpy()
    aligned_future = encode_with_history_torch(history_tensor, future_tensor, "heading_aligned").numpy()

    def grouped(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values[:, 0, :], values[:, 1:, 0], values[:, 1:, 1]], axis=1)

    np.testing.assert_array_equal(encode_positions_numpy(aligned_history, "displacement").astype(np.float32), grouped(online_history))
    np.testing.assert_array_equal(encode_positions_numpy(aligned_future, "displacement").astype(np.float32), grouped(online_future))


def _trajectory(offset: float = 0.0, curve: float = 0.02) -> np.ndarray:
    steps = np.arange(30, dtype=np.float64)
    return np.stack([offset + 1.7 * steps, -offset + 0.4 * steps + curve * steps**2], axis=-1)


def _write_raw_dataset(root: Path, *, validation_offset: float = 0.0, test_offset: float = 0.0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    counts = {"train": 4, "val": 2, "test": 2}
    offsets = {"train": 0.0, "val": validation_offset, "test": test_offset}
    for split, count in counts.items():
        rows = []
        for index in range(count):
            base = offsets[split] + index * 10.0
            history = _trajectory(base)
            future_steps = np.arange(30, 60, dtype=np.float64)
            future = np.stack(
                [base + 1.7 * future_steps, -base + 0.4 * future_steps + 0.02 * future_steps**2],
                axis=-1,
            )
            rows.append(
                {
                    "sample_id": f"{split}-{index}",
                    "hist_x_json": json.dumps(history[:, 0].tolist()),
                    "hist_y_json": json.dumps(history[:, 1].tolist()),
                    "fut_x_json": json.dumps(future[:, 0].tolist()),
                    "fut_y_json": json.dumps(future[:, 1].tolist()),
                }
            )
        split_dir = root / split
        split_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(split_dir / "part-000.csv", index=False)
    (root / "summary.json").write_text(json.dumps({"identity": root.name}))


def _write_processed_dataset(root: Path, *, validation_offset: float = 0.0, test_offset: float = 0.0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    counts = {"train": 4, "val": 2, "test": 2}
    all_charts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, count in counts.items():
        split_offset = 0.0 if split == "train" else validation_offset if split == "val" else test_offset
        histories = []
        futures = []
        for index in range(count):
            history = _trajectory(split_offset + index * 5.0).astype(np.float32)
            last_delta = history[-1] - history[-2]
            steps = np.arange(1, 31, dtype=np.float32)[:, None]
            future = history[-1:] + steps * last_delta
            angle = np.arctan2(last_delta[1], last_delta[0])
            rotation = np.asarray(
                [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]],
                dtype=np.float32,
            )
            origin = history[-1].copy()
            history = (history - origin) @ rotation.T
            future = (future - origin) @ rotation.T
            histories.append(history)
            futures.append(future)
        history_array = np.stack(histories)
        future_array = np.stack(futures)
        history_chart = encode_positions_numpy(history_array, chart_type="displacement").astype(np.float32)
        future_chart = encode_positions_numpy(future_array, chart_type="displacement").astype(np.float32)
        all_charts[split] = (history_chart, future_chart)
        np.savez_compressed(
            root / f"{split}.npz",
            sample_id=np.asarray([f"{split}-{index}" for index in range(count)]),
            history=history_array,
            future=future_array,
            history_chart=history_chart,
            future_chart=future_chart,
        )
    train_history_chart, train_future_chart = all_charts["train"]
    history_std = np.where(train_history_chart.std(axis=0) < 1e-6, 1.0, train_history_chart.std(axis=0))
    future_std = np.where(train_future_chart.std(axis=0) < 1e-6, 1.0, train_future_chart.std(axis=0))
    (root / "normalization.json").write_text(
        json.dumps(
            {
                "history_chart_mean": train_history_chart.mean(axis=0).tolist(),
                "history_chart_std": history_std.tolist(),
                "future_chart_mean": train_future_chart.mean(axis=0).tolist(),
                "future_chart_std": future_std.tolist(),
            }
        )
    )
    (root / "feasibility_thresholds.json").write_text(json.dumps({"max_acceleration": 1e9, "max_turn_rate": 1e9}))
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": "synthetic-section3-smoke",
                "split_sizes": counts,
                "chart_type": "displacement",
                "align_to_last_heading": True,
                "history_steps": 30,
                "future_steps": 30,
                "dt_seconds": 20.0,
                "coordinate_unit": "metres",
            }
        )
    )


def _config(tmp_path: Path, processed_root: Path, model_name: str, *, seed: int = 0) -> dict:
    loss_name = "kineroute" if model_name in CHART_MODELS else "mse"
    args = parse_args(
        [
            "--experiment-name",
            "section3_smoke",
            "--run-name",
            f"{model_name}_{processed_root.name}_{seed}",
            "--processed-root",
            str(processed_root),
            "--results-root",
            str(tmp_path / "results"),
            "--model-name",
            model_name,
            "--seed",
            str(seed),
            "--hidden-dim",
            "8",
            "--d-model",
            "8",
            "--num-heads",
            "2",
            "--num-layers",
            "1",
            "--hidden-layers",
            "1",
            "--residual-blocks",
            "1",
            "--levels",
            "1",
            "--num-blocks",
            "1",
            "--num-routed-blocks",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--checkpoint-every-epochs",
            "1",
            "--device",
            "cpu",
            "--preprocessing-mode",
            "heading_aligned_displacement",
            "--loss-name",
            loss_name,
            "--lambda-velocity",
            "0",
            "--lambda-acceleration",
            "0",
            "--lambda-turn",
            "0",
            "--lambda-feasibility",
            "0",
            "--save-prediction-examples",
            "1",
        ]
    )
    return args_to_config(args)


def test_frozen_compact_manifest_matches_sources_and_has_no_overlap() -> None:
    manifest = load_split_manifest(COMPACT_MANIFEST)

    assert manifest["policy"] == "one frozen compact train/val/test split shared by all models and all five training seeds"
    assert manifest["training_seeds"] == [0, 1, 2, 3, 4]
    assert manifest["split_counts"] == {"train": 29635, "val": 3667, "test": 3262}
    validate_raw_sources(COMPACT_RAW_ROOTS, manifest)

    observed: dict[str, list[str]] = {split: [] for split in SPLITS}
    for root in COMPACT_RAW_ROOTS:
        for split in SPLITS:
            path = root / split / "part-000.csv"
            if not path.exists():
                path = root / split / "part-000.csv.gz"
            coordinate_columns = ["hist_x_json", "hist_y_json", "fut_x_json", "fut_y_json"]
            frame = pd.read_csv(path, usecols=["sample_id", *coordinate_columns])
            raw_ids = frame["sample_id"].astype(str)
            observed[split].extend(f"{root.name}::{sample_id}" for sample_id in raw_ids)
            assert frame["hist_x_json"].map(lambda value: len(json.loads(value)) == 30).all()
            assert frame["hist_y_json"].map(lambda value: len(json.loads(value)) == 30).all()
            assert frame["fut_x_json"].map(lambda value: len(json.loads(value)) == 30).all()
            assert frame["fut_y_json"].map(lambda value: len(json.loads(value)) == 30).all()
    for split in SPLITS:
        validate_split_membership(split, observed[split], manifest)
        assert sample_ids_fingerprint(observed[split]) == manifest["split_fingerprints"][split]
    assert set(observed["train"]).isdisjoint(observed["val"])
    assert set(observed["train"]).isdisjoint(observed["test"])
    assert set(observed["val"]).isdisjoint(observed["test"])
    assert all(file["protocol_shapes_verified"] for source in manifest["sources"] for file in source["files"].values())


def test_coordinate_units_and_sampling_interval_are_tied_to_hashed_source_evidence() -> None:
    manifest = load_split_manifest(COMPACT_MANIFEST)
    protocol = manifest["protocol"]

    assert protocol["coordinate_unit"] == "metres"
    assert protocol["coordinate_frame"] == "local planar anchor-relative frame; x=east, y=north"
    assert protocol["dt_seconds"] == 20.0
    assert protocol["history_steps"] == 30
    assert protocol["future_steps"] == 30
    assert protocol["split_membership_stage"] == "before model-specific preprocessing and normalization"
    assert protocol["model_preprocessing_modes"] == ["raw", "heading_aligned", "heading_aligned_displacement"]
    for evidence_key in ("coordinate_unit_evidence", "sampling_interval_evidence"):
        evidence = protocol[evidence_key]
        evidence_path = REPO_ROOT / evidence["path"]
        assert evidence_path.exists()
        assert sha256_file(evidence_path) == evidence["sha256"]
    parsed = validate_manifest_evidence(manifest, REPO_ROOT)
    assert parsed["coordinate_unit"] == "metres"
    assert parsed["coordinate_frame"] == protocol["coordinate_frame"]
    assert parsed["dt_seconds"] == 20.0
    assert parsed["history_steps"] == 30
    assert parsed["future_steps"] == 30


def test_evidence_hash_alone_cannot_validate_an_unsupported_statement(tmp_path: Path) -> None:
    data_card = tmp_path / "DATA_CARD.md"
    readme = tmp_path / "README.md"
    data_card.write_text(
        "All positions are in a local planar frame centred on the anchor (last history point), "
        "with x = east, y = north, units of metres.\n"
    )
    readme.write_text("- Sampling interval: 20 seconds\n- History length: 30 points\n- Future length: 30 points\n")
    manifest = load_split_manifest(COMPACT_MANIFEST)
    manifest["protocol"]["coordinate_unit_evidence"] = {
        "path": "DATA_CARD.md",
        "sha256": sha256_file(data_card),
        "statement": "Coordinates are longitude and latitude in degrees.",
    }

    with pytest.raises(ValueError, match="evidence statement does not match"):
        validate_manifest_evidence(manifest, tmp_path)


def test_five_training_seeds_resolve_to_one_identical_split() -> None:
    configs = []
    for seed in (0, 1):
        args = parse_args(["--config", "experiments/baselines/gru_coords_paper.yaml", "--seed", str(seed)])
        configs.append(args_to_config(args))

    assert configs[0]["seed"] == 0
    assert configs[1]["seed"] == 1
    assert configs[0]["data"] == configs[1]["data"]
    assert configs[0]["data"]["split_manifest"] == "experiments/compact/split_manifest.json"
    assert [configs[0]["data"][f"{name}_split"] for name in SPLITS] == list(SPLITS)


def test_manifest_backed_preparation_is_repeatable_and_rejects_source_drift(tmp_path: Path) -> None:
    raw_root = tmp_path / "tiny_source"
    _write_raw_dataset(raw_root)
    data_card = tmp_path / "DATA_CARD.md"
    readme = tmp_path / "README.md"
    data_card.write_text(
        "All positions are in a local planar frame centred on the anchor (last history point), "
        "with x = east, y = north, units of metres.\n"
    )
    readme.write_text("- Sampling interval: 20 seconds\n- History length: 30 points\n- Future length: 30 points\n")
    evidence = {
        "path": "DATA_CARD.md",
        "sha256": sha256_file(data_card),
        "statement": "All positions use a local planar frame centred on the last history point; x=east, y=north, in metres.",
    }
    interval_evidence = {
        "path": "README.md",
        "sha256": sha256_file(readme),
        "statement": "Sampling interval: 20 seconds; history length: 30 points; future length: 30 points.",
    }
    manifest = create_split_manifest(
        raw_roots=[raw_root],
        repo_root=tmp_path,
        dataset_id="tiny-frozen",
        coordinate_unit="metres",
        coordinate_frame="local planar anchor-relative frame; x=east, y=north",
        coordinate_unit_evidence=evidence,
        sampling_interval_evidence=interval_evidence,
    )
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    first = prepare_envship_dataset(raw_root, tmp_path / "processed_first", 20.0, 0.995, chart_type="displacement", split_manifest_path=manifest_path)
    second = prepare_envship_dataset(raw_root, tmp_path / "processed_second", 20.0, 0.995, chart_type="displacement", split_manifest_path=manifest_path)
    for split in SPLITS:
        with np.load(first.processed_root / f"{split}.npz", allow_pickle=True) as first_payload, np.load(
            second.processed_root / f"{split}.npz", allow_pickle=True
        ) as second_payload:
            assert first_payload["sample_id"].tolist() == second_payload["sample_id"].tolist()
            validate_split_membership(split, first_payload["sample_id"].astype(str).tolist(), manifest)
    validate_processed_dataset(first.processed_root, manifest_path)
    train_path = first.processed_root / "train.npz"
    with np.load(train_path, allow_pickle=True) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["history"][0, 0, 0] += 1.0
    np.savez_compressed(train_path, **arrays)
    with pytest.raises(ValueError, match="semantic fingerprint mismatch"):
        validate_processed_dataset(first.processed_root, manifest_path)

    changed = pd.read_csv(raw_root / "test/part-000.csv")
    changed.loc[0, "sample_id"] = "unexpected-repartition"
    changed.to_csv(raw_root / "test/part-000.csv", index=False)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        prepare_envship_dataset(raw_root, tmp_path / "processed_changed", 20.0, 0.995, chart_type="displacement", split_manifest_path=manifest_path)


@pytest.mark.parametrize("mode", ["raw", "heading_aligned", "heading_aligned_displacement"])
@pytest.mark.parametrize(
    "trajectory",
    [
        _trajectory(),
        np.repeat(np.asarray([[12.0, -7.0]], dtype=np.float64), 30, axis=0),
        np.stack([np.linspace(0.0, 1e-9, 30), np.linspace(0.0, -1e-9, 30)], axis=-1),
    ],
    ids=["curved", "stationary", "near_stationary"],
)
def test_all_preprocessing_modes_round_trip_nontrivial_and_stationary_cases(mode: str, trajectory: np.ndarray) -> None:
    encoded, transform = encode_trajectory(trajectory, mode)
    np.testing.assert_allclose(transform.decode(encoded), trajectory, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("mode", ["raw", "heading_aligned", "heading_aligned_displacement"])
def test_history_fitted_transform_round_trips_future_and_has_no_leakage(mode: str) -> None:
    history = _trajectory(offset=5.0)
    future_a = _trajectory(offset=100.0, curve=0.04)
    future_b = _trajectory(offset=-700.0, curve=-0.03)

    encoded_history_a, encoded_future_a, transform_a = encode_history_future(history, future_a, mode)
    encoded_history_b, _, transform_b = encode_history_future(history, future_b, mode)

    np.testing.assert_allclose(transform_a.origin, transform_b.origin)
    assert transform_a.heading == transform_b.heading
    np.testing.assert_allclose(encoded_history_a, encoded_history_b)
    np.testing.assert_allclose(transform_a.decode(encoded_future_a), future_a, atol=1e-5)

    history_tensor = torch.from_numpy(history).float().unsqueeze(0)
    future_a_tensor = torch.from_numpy(future_a).float().unsqueeze(0)
    future_b_tensor = torch.from_numpy(future_b).float().unsqueeze(0)
    model_input_a = encode_with_history_torch(history_tensor, history_tensor, mode)
    model_input_b = encode_with_history_torch(history_tensor, history_tensor, mode)
    torch.testing.assert_close(model_input_a, model_input_b)
    encoded_future = encode_with_history_torch(history_tensor, future_a_tensor, mode)
    torch.testing.assert_close(decode_with_history_torch(history_tensor, encoded_future, mode), future_a_tensor, atol=1e-4, rtol=1e-5)
    assert not torch.equal(future_a_tensor, future_b_tensor)


def test_changing_future_cannot_change_displacement_input_seen_by_forecaster(tmp_path: Path) -> None:
    class CaptureForecaster(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: torch.Tensor | None = None

        def forward(self, history: torch.Tensor) -> torch.Tensor:
            self.seen = history.detach().clone()
            return torch.zeros_like(history)

    processed_root = tmp_path / "processed"
    _write_processed_dataset(processed_root)
    runner = ExperimentRunner(_config(tmp_path, processed_root, "gru"), "section3", REPO_ROOT, "test")
    batch = next(iter(DataLoader(ProcessedTrajectoryDataset(processed_root, "train"), batch_size=2)))
    changed_future_batch = dict(batch)
    changed_future_batch["future"] = batch["future"] + 100000.0
    first_model = CaptureForecaster()
    second_model = CaptureForecaster()

    runner._predict_positions(first_model, batch, train=True, position_norm=None, chart_norm=None, chart_type="displacement")
    runner._predict_positions(second_model, changed_future_batch, train=True, position_norm=None, chart_norm=None, chart_type="displacement")

    assert first_model.seen is not None and second_model.seen is not None
    torch.testing.assert_close(first_model.seen, second_model.seen)


def test_dataset_and_config_fail_loudly_on_protocol_violations(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    _write_processed_dataset(processed_root)
    original = np.load(processed_root / "train.npz", allow_pickle=True)
    payload = {key: original[key] for key in original.files}
    original.close()
    payload["history"] = payload["history"][:, :29, :]
    np.savez_compressed(processed_root / "train.npz", **payload)
    with pytest.raises(ValueError, match=r"\[N, 30, 2\]"):
        ProcessedTrajectoryDataset(processed_root, "train")

    config = _config(tmp_path, processed_root, "gru")
    config["data"]["history_steps"] = 29
    metadata = json.loads((processed_root / "metadata.json").read_text())
    with pytest.raises(ValueError, match="history_steps must be 30"):
        validate_experiment_protocol(config, metadata)


def test_chart_model_rejects_processed_preprocessing_mismatch(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    _write_processed_dataset(processed_root)
    config = _config(tmp_path, processed_root, "kineroute_nvp")
    metadata = json.loads((processed_root / "metadata.json").read_text())
    metadata["align_to_last_heading"] = False
    metadata["chart_type"] = "position"
    config["data"]["align_to_last_heading"] = True
    config["data"]["chart_type"] = "displacement"

    with pytest.raises(ValueError, match="Processed chart preprocessing disagrees"):
        validate_experiment_protocol(config, metadata)


def test_normalization_and_feasibility_statistics_depend_only_on_training_split(tmp_path: Path) -> None:
    raw_a = tmp_path / "raw_a"
    raw_b = tmp_path / "raw_b"
    _write_raw_dataset(raw_a, validation_offset=0.0, test_offset=0.0)
    _write_raw_dataset(raw_b, validation_offset=100000.0, test_offset=-100000.0)
    processed_a = tmp_path / "processed_a"
    processed_b = tmp_path / "processed_b"
    prepare_envship_dataset(raw_a, processed_a, 20.0, 0.995, chart_type="displacement")
    prepare_envship_dataset(raw_b, processed_b, 20.0, 0.995, chart_type="displacement")

    assert json.loads((processed_a / "normalization.json").read_text()) == json.loads((processed_b / "normalization.json").read_text())
    assert json.loads((processed_a / "feasibility_thresholds.json").read_text()) == json.loads(
        (processed_b / "feasibility_thresholds.json").read_text()
    )

    runner_a = ExperimentRunner(_config(tmp_path / "a", processed_a, "gru"), "section3", REPO_ROOT, "test")
    runner_b = ExperimentRunner(_config(tmp_path / "b", processed_b, "gru"), "section3", REPO_ROOT, "test")
    train_a, _, _ = runner_a._build_loaders(processed_a)
    train_b, _, _ = runner_b._build_loaders(processed_b)
    norm_a = runner_a._position_normalization(train_a)
    norm_b = runner_b._position_normalization(train_b)
    assert norm_a is not None and norm_b is not None
    for left, right in zip(norm_a, norm_b):
        torch.testing.assert_close(left, right)
    assert norm_a[0].shape == (1, 30, 2)
    encoded_train = encode_with_history_torch(
        torch.as_tensor(train_a.dataset.history),
        torch.as_tensor(train_a.dataset.history),
        "heading_aligned_displacement",
    )
    torch.testing.assert_close(norm_a[0], encoded_train.mean(dim=0, keepdim=True))
    torch.testing.assert_close(norm_a[1], encoded_train.std(dim=0, keepdim=True, correction=0).clamp_min(1e-6))


@pytest.mark.parametrize(
    "model_name",
    [
        "dead_reckoning",
        "gru",
        "lstm",
        "mlp",
        "resnet",
        "tcn",
        "transformer_nar",
        "realnvp",
        "kineroute_nvp",
        "non_invertible_routed",
    ],
)
def test_every_model_family_end_to_end_through_shared_system(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_name: str) -> None:
    processed_root = tmp_path / "processed"
    _write_processed_dataset(processed_root)
    config = _config(tmp_path, processed_root, model_name)
    runner = ExperimentRunner(config, "section3_smoke", REPO_ROOT, "test")
    summary = runner._prepare_data()
    train_loader, _, test_loader = runner._build_loaders(summary.processed_root)
    model = build_model(config).to(runner.device)
    trainable = count_trainable_parameters(model) > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) if trainable else None
    position_norm = runner._position_normalization(train_loader)
    chart_norm = runner._chart_normalization(summary) if runner.uses_chart_model else None
    thresholds = json.loads(summary.thresholds_path.read_text())

    before = [parameter.detach().clone() for parameter in model.parameters()] if trainable else []
    train_metrics = runner._run_epoch(
        model,
        train_loader,
        optimizer,
        thresholds,
        position_norm,
        chart_norm,
        "displacement",
        train=trainable,
    )
    assert all(np.isfinite(value) for value in train_metrics.values())
    if trainable:
        assert any(not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters()))

    checkpoint_path = runner.artifacts.save_checkpoint(
        "smoke.pt", model=model, optimizer=optimizer, scheduler=None, epoch=1 if trainable else 0
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["optimizer_state"] is not None if trainable else checkpoint["optimizer_state"] is None
    restored = build_model(config).to(runner.device)
    restored.load_state_dict(checkpoint["model_state"])

    shared_metric_calls = []
    real_summarize = trainer_module.summarize_predictions

    def tracked_summarize(*args, **kwargs):
        shared_metric_calls.append(model_name)
        return real_summarize(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "summarize_predictions", tracked_summarize)
    metrics, _ = runner.evaluate_split(restored, test_loader, thresholds, position_norm, chart_norm, "displacement")
    assert shared_metric_calls == [model_name]
    assert np.isfinite(metrics["ade"])
    assert np.isfinite(metrics["fde"])

    if model_name in {"realnvp", "kineroute_nvp"}:
        chart = torch.randn(3, 60)
        torch.testing.assert_close(restored.inverse(restored(chart)), chart, atol=1e-5, rtol=1e-5)
    if model_name == "non_invertible_routed":
        assert not hasattr(restored, "inverse")
