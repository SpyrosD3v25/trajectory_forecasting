from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kineroute_nvp.models.factory import build_model
from kineroute_nvp.utils.config import args_to_config, parse_args


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "experiments/preprocessing_ablation"


EXPECTED_MATRIX = {
    ("gru", "raw", "mse"),
    ("lstm", "raw", "mse"),
    ("gru", "heading_aligned", "mse"),
    ("lstm", "heading_aligned", "mse"),
    ("gru", "heading_aligned_displacement", "mse"),
    ("lstm", "heading_aligned_displacement", "mse"),
    ("gru", "heading_aligned_displacement", "ade"),
    ("lstm", "heading_aligned_displacement", "ade"),
}


def test_section4_has_exact_gpu_config_matrix_on_one_frozen_split() -> None:
    config_paths = sorted(CONFIG_ROOT.glob("*.yaml"))
    assert len(config_paths) == 8
    observed = set()
    membership_signatures = set()
    for path in config_paths:
        config = args_to_config(parse_args(["--config", str(path.relative_to(REPO_ROOT))]))
        observed.add((config["model"]["name"], config["data"]["preprocessing_mode"], config["loss"]["name"]))
        membership_signatures.add(
            (
                tuple(config["data"]["raw_roots"]),
                config["data"]["processed_root"],
                config["data"]["split_manifest"],
                config["data"]["train_split"],
                config["data"]["val_split"],
                config["data"]["test_split"],
                config["data"]["history_steps"],
                config["data"]["future_steps"],
                config["data"]["dt_seconds"],
            )
        )
        assert config["seed"] == 0
        assert config["training"]["device"] == "cuda"
        assert config["evaluation"]["evaluate_test"] is False
        assert config["data"]["prepare_if_missing"] is False
        assert config["model"]["hidden_dim"] == 128
        assert config["model"]["num_layers"] == 2
        assert config["loss"]["normalize"] is True
        assert build_model(config).__class__.__name__ in {"GRUBaseline", "LSTMBaseline"}
    assert observed == EXPECTED_MATRIX
    assert len(membership_signatures) == 1
    signature = next(iter(membership_signatures))
    assert signature[2] == "experiments/compact/split_manifest.json"
    assert signature[3:6] == ("train", "val", "test")
    assert signature[6:] == (30, 30, 20.0)

    launcher = (CONFIG_ROOT / "run_all.sh").read_text()
    assert "preflight_compact_data.py" in launcher
    assert "--repair" not in launcher


def _write_result(
    root: Path,
    name: str,
    *,
    experiment: str,
    model: str,
    mode: str,
    loss: str,
    seed: int,
    status: str = "completed",
    learning_rate: float = 0.001,
    manifest_sha256: str = "manifest-sha",
) -> Path:
    result_dir = root / name
    result_dir.mkdir(parents=True)
    identity = {
        "manifest_path": "experiments/compact/split_manifest.json",
        "manifest_sha256": manifest_sha256,
        "dataset_id": "compact",
        "dataset_sha256": "dataset-sha",
        "split_counts": {"train": 29635, "val": 3667, "test": 3262},
        "split_fingerprints": {"train": "a", "val": "b", "test": "c"},
        "processed_split_fingerprints": {"train": "pa", "val": "pb", "test": "pc"},
        "processed_chart_type": "position",
        "processed_align_to_last_heading": False,
        "protocol": {"history_steps": 30, "future_steps": 30, "dt_seconds": 20.0},
    }
    (result_dir / "_results.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": 2,
                "experiment_name": experiment,
                "run_name": name,
                "status": status,
                "seed": seed,
                "model": model,
                "model_parameter_count": 1234,
                "best_validation_ade": 10.0,
                "best_validation_fde": 20.0,
                "test_ade": None,
                "test_fde": None,
                "test_evaluated": False,
                "runtime_source_identity": {"sha256": "source-sha", "file_count": 1, "files": ["src/model.py"]},
                "dataset_identity": identity,
                "dataset_split_sizes": identity["split_counts"],
                "resolved_configuration": {
                    "run_name": name,
                    "seed": seed,
                    "data": {
                        "raw_roots": ["raw-a", "raw-b"],
                        "processed_root": "processed",
                        "split_manifest": "experiments/compact/split_manifest.json",
                        "train_split": "train",
                        "val_split": "val",
                        "test_split": "test",
                        "history_steps": 30,
                        "future_steps": 30,
                        "dt_seconds": 20.0,
                        "prepare_if_missing": False,
                        "preprocessing_mode": mode,
                        "align_to_last_heading": mode != "raw",
                        "chart_type": "displacement" if mode == "heading_aligned_displacement" else "position",
                    },
                    "model": {"name": model, "hidden_dim": 128, "num_layers": 2},
                    "loss": {"name": loss, "normalize": True},
                    "optimizer": {"name": "adamw", "learning_rate": learning_rate, "weight_decay": 0.0001},
                    "scheduler": {"name": "reduce_on_plateau", "factor": 0.5, "patience": 8},
                    "training": {"epochs": 80, "batch_size": 256, "early_stopping_patience": 15},
                },
            }
        )
    )
    return result_dir / "_results.json"


def test_preprocessing_ablation_aggregator_emits_attribution_columns(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_result(results_root, "gru_raw", experiment="preprocessing_ablation", model="gru", mode="raw", loss="mse", seed=0)
    _write_result(
        results_root,
        "lstm_disp",
        experiment="preprocessing_ablation",
        model="lstm",
        mode="heading_aligned_displacement",
        loss="ade",
        seed=0,
    )
    _write_result(results_root, "unrelated", experiment="baselines", model="gru", mode="raw", loss="mse", seed=0)
    output = tmp_path / "table.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_preprocessing_ablation.py",
            "--results-root",
            str(results_root),
            "--output",
            str(output),
            "--allow-partial",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert set(rows[0]) >= {
        "model",
        "alignment",
        "displacement",
        "loss",
        "params",
        "best_validation_ade",
        "best_validation_fde",
        "test_ade",
        "test_fde",
    }
    by_run = {row["run_name"]: row for row in rows}
    assert by_run["gru_raw"]["alignment"] == "False"
    assert by_run["gru_raw"]["displacement"] == "False"
    assert by_run["lstm_disp"]["alignment"] == "True"
    assert by_run["lstm_disp"]["displacement"] == "True"


def _aggregate(results_root: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_preprocessing_ablation.py",
            "--results-root",
            str(results_root),
            "--output",
            str(output),
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_preprocessing_aggregator_requires_exact_eight_cell_matrix(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    for model, mode, loss in EXPECTED_MATRIX:
        _write_result(
            results_root,
            f"{model}_{mode}_{loss}",
            experiment="preprocessing_ablation",
            model=model,
            mode=mode,
            loss=loss,
            seed=0,
        )
    output = tmp_path / "table.csv"

    completed = _aggregate(results_root, output)

    assert completed.returncode == 0, completed.stderr
    with output.open() as handle:
        assert len(list(csv.DictReader(handle))) == 8
    selection = json.loads((tmp_path / "table_selection.json").read_text())
    assert selection["test_metrics_used_for_selection"] is False
    assert selection["selected"]["preprocessing_mode"] == "raw"
    assert selection["selected"]["loss"] == "mse"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("mixed_seed", "expected the single attribution seed"),
        ("mixed_manifest", "different dataset/manifest identities"),
        ("different_optimizer", "controlled configuration fields"),
        ("failed", "requires status=completed"),
        ("non_finite", "must be finite"),
    ],
)
def test_preprocessing_aggregator_rejects_invalid_science(tmp_path: Path, mutation: str, message: str) -> None:
    results_root = tmp_path / "results"
    first = _write_result(
        results_root,
        "gru_raw",
        experiment="preprocessing_ablation",
        model="gru",
        mode="raw",
        loss="mse",
        seed=0,
    )
    second = _write_result(
        results_root,
        "gru_aligned",
        experiment="preprocessing_ablation",
        model="gru",
        mode="heading_aligned",
        loss="mse",
        seed=1 if mutation == "mixed_seed" else 0,
        manifest_sha256="other" if mutation == "mixed_manifest" else "manifest-sha",
        learning_rate=0.01 if mutation == "different_optimizer" else 0.001,
        status="failed" if mutation == "failed" else "completed",
    )
    if mutation == "non_finite":
        payload = json.loads(second.read_text())
        payload["best_validation_ade"] = float("nan")
        second.write_text(json.dumps(payload))

    completed = _aggregate(results_root, tmp_path / "table.csv", "--allow-partial")

    assert completed.returncode != 0
    assert message in completed.stderr
    assert first.exists()


def test_preprocessing_aggregator_rejects_duplicate_logical_cell(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    for name in ("first", "duplicate"):
        _write_result(
            results_root,
            name,
            experiment="preprocessing_ablation",
            model="gru",
            mode="raw",
            loss="mse",
            seed=0,
        )

    completed = _aggregate(results_root, tmp_path / "table.csv", "--allow-partial")

    assert completed.returncode != 0
    assert "Duplicate preprocessing-ablation cell" in completed.stderr
