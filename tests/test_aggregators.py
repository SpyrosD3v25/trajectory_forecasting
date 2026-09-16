from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload(seed: int, *, learning_rate: float = 0.001, identity: bool = True) -> dict:
    config = {
        "run_name": f"model_seed_{seed}",
        "seed": seed,
        "data": {
            "preprocessing_mode": "heading_aligned_displacement",
            "split_manifest": "experiments/compact/split_manifest.json",
            "processed_root": "processed",
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
        },
        "model": {"name": "gru", "hidden_dim": 128, "num_layers": 2},
        "loss": {"name": "ade", "normalize": True},
        "optimizer": {"learning_rate": learning_rate, "weight_decay": 0.0001},
        "training": {"epochs": 80, "batch_size": 256},
        "output": {"results_root": "results"},
    }
    payload = {
        "artifact_schema_version": 2,
        "status": "completed",
        "experiment_name": "multiseed",
        "run_name": config["run_name"],
        "seed": seed,
        "resolved_configuration": config,
        "best_validation_ade": 10.0 + seed,
        "best_validation_fde": 20.0 + seed,
        "test_ade": 11.0 + seed,
        "test_fde": 21.0 + seed,
        "runtime_source_identity": {"sha256": "source", "file_count": 1, "files": ["src/model.py"]},
    }
    if identity:
        payload["dataset_identity"] = {
            "manifest_sha256": "manifest",
            "dataset_sha256": "dataset",
            "split_fingerprints": {"train": "a", "val": "b", "test": "c"},
        }
    return payload


def _write(root: Path, name: str, payload: dict) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "_results.json").write_text(json.dumps(payload))


def _run(root: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_multiseed.py",
            "--results-root",
            str(root),
            "--output",
            str(output),
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_multiseed_aggregator_requires_one_exact_definition_and_five_seeds(tmp_path: Path) -> None:
    root = tmp_path / "results"
    for seed in range(5):
        _write(root, f"seed_{seed}", _payload(seed))

    output = tmp_path / "aggregate.csv"
    completed = _run(root, output)

    assert completed.returncode == 0, completed.stderr
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["seeds"] == "0 1 2 3 4"
    assert rows[0]["runs"] == "5"
    assert rows[0]["manifest_sha256"] == "manifest"


def test_multiseed_aggregator_does_not_merge_changed_hyperparameters(tmp_path: Path) -> None:
    root = tmp_path / "results"
    for seed in range(5):
        _write(root, f"seed_{seed}", _payload(seed, learning_rate=0.01 if seed == 4 else 0.001))

    completed = _run(root, tmp_path / "aggregate.csv")

    assert completed.returncode != 0
    assert "Incomplete seed set" in completed.stderr


def test_multiseed_aggregator_rejects_duplicate_seed_and_missing_identity(tmp_path: Path) -> None:
    duplicate_root = tmp_path / "duplicate"
    _write(duplicate_root, "first", _payload(0))
    _write(duplicate_root, "second", _payload(0))
    duplicate = _run(duplicate_root, tmp_path / "duplicate.csv", "--allow-partial")
    assert duplicate.returncode != 0
    assert "Duplicate seed" in duplicate.stderr

    unfrozen_root = tmp_path / "unfrozen"
    _write(unfrozen_root, "seed_0", _payload(0, identity=False))
    unfrozen = _run(unfrozen_root, tmp_path / "unfrozen.csv", "--allow-partial")
    assert unfrozen.returncode != 0
    assert "missing dataset_identity" in unfrozen.stderr


def test_multiseed_aggregator_does_not_merge_runtime_source_changes(tmp_path: Path) -> None:
    root = tmp_path / "results"
    for seed in range(5):
        payload = _payload(seed)
        if seed == 4:
            payload["runtime_source_identity"]["sha256"] = "changed-source"
        _write(root, f"seed_{seed}", payload)

    completed = _run(root, tmp_path / "aggregate.csv")

    assert completed.returncode != 0
    assert "Incomplete seed set" in completed.stderr
