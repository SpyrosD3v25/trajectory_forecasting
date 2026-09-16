import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from kineroute_nvp.data.dataset import ProcessedTrajectoryDataset
from kineroute_nvp.data.trajectory_preprocessing import encode_with_history_torch
from kineroute_nvp.losses.trajectory_physics import trajectory_loss
from kineroute_nvp.models.baselines import DeadReckoningBaseline
from kineroute_nvp.training.trainer import ExperimentRunner
from kineroute_nvp.utils.config import args_to_config, parse_args


def _write_processed_split(root: Path, split: str, samples: int) -> None:
    history = np.zeros((samples, 30, 2), dtype=np.float32)
    future = np.zeros((samples, 30, 2), dtype=np.float32)
    for index in range(samples):
        speed = 0.1 * (index + 1)
        history[index, :, 0] = np.arange(30, dtype=np.float32) * speed
        future[index, :, 0] = np.arange(30, 60, dtype=np.float32) * speed
    chart = np.zeros((samples, 60), dtype=np.float32)
    np.savez_compressed(
        root / f"{split}.npz",
        sample_id=np.asarray([f"{split}-{index}" for index in range(samples)]),
        history=history,
        future=future,
        history_chart=chart,
        future_chart=chart,
    )


def _write_processed_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    sizes = {"train": 8, "val": 4, "test": 4}
    for split, samples in sizes.items():
        _write_processed_split(root, split, samples)
    (root / "metadata.json").write_text(json.dumps({"split_sizes": sizes, "chart_type": "position"}))
    normalization = {
        "history_chart_mean": [0.0] * 60,
        "history_chart_std": [1.0] * 60,
        "future_chart_mean": [0.0] * 60,
        "future_chart_std": [1.0] * 60,
    }
    (root / "normalization.json").write_text(json.dumps(normalization))


def _run_train(repo_root: Path, processed_root: Path, results_root: Path, model_name: str, run_name: str) -> Path:
    command = [
        sys.executable,
        "scripts/train.py",
        "--experiment-name",
        "unified",
        "--run-name",
        run_name,
        "--processed-root",
        str(processed_root),
        "--results-root",
        str(results_root),
        "--model-name",
        model_name,
        "--hidden-dim",
        "8",
        "--num-layers",
        "1",
        "--hidden-layers",
        "1",
        "--num-routed-blocks",
        "1",
        "--epochs",
        "2",
        "--batch-size",
        "4",
        "--num-workers",
        "0",
        "--checkpoint-every-epochs",
        "1",
        "--device",
        "cpu",
        "--preprocessing-mode",
        "raw",
        "--loss-name",
        "kineroute" if model_name == "kineroute_nvp" else "mse",
    ]
    if model_name == "dead_reckoning":
        command.extend(["--loss-name", "mse"])
    completed = subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)
    return Path(completed.stdout.strip().splitlines()[-1])


def _config(tmp_path: Path, processed_root: Path, model_name: str = "gru", preprocessing_mode: str = "raw") -> dict:
    args = parse_args(
        [
            "--experiment-name",
            "unit",
            "--run-name",
            f"{model_name}_{preprocessing_mode}",
            "--processed-root",
            str(processed_root),
            "--results-root",
            str(tmp_path / "results"),
            "--model-name",
            model_name,
            "--hidden-dim",
            "8",
            "--num-layers",
            "1",
            "--hidden-layers",
            "1",
            "--num-routed-blocks",
            "1",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--checkpoint-every-epochs",
            "1",
            "--device",
            "cpu",
            "--preprocessing-mode",
            preprocessing_mode,
            "--loss-name",
            "kineroute" if model_name == "kineroute_nvp" else "mse",
        ]
    )
    return args_to_config(args)


class CaptureForecaster(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.Tensor | None = None

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        self.seen = history.detach().clone()
        return torch.zeros_like(history)


def test_config_precedence_and_preprocessing_mode() -> None:
    args = parse_args(
        [
            "--config",
            "experiments/baselines/gru_coords_paper.yaml",
            "--hidden-dim",
            "64",
            "--preprocessing-mode",
            "heading_aligned_displacement",
        ]
    )
    config = args_to_config(args)
    assert config["model"]["hidden_dim"] == 64
    assert config["data"]["preprocessing_mode"] == "heading_aligned_displacement"
    assert config["data"]["align_to_last_heading"] is True
    assert config["data"]["chart_type"] == "displacement"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--model-name", "unknown"], "Unsupported model"),
        (["--model-name", "transformer_nar", "--d-model", "10", "--num-heads", "4"], "divisible"),
        (["--learning-rate", "0"], "learning_rate"),
        (["--teacher-forcing-ratio", "0.5", "--model-name", "gru"], "only implemented"),
        (["--lambda-reverse", "1", "--model-name", "gru"], "invertible chart model"),
        (["--max-train-samples", "0"], "must be positive"),
    ],
)
def test_invalid_resolved_configs_fail_loudly(arguments: list[str], message: str) -> None:
    base = ["--experiment-name", "unit", "--run-name", "invalid", "--processed-root", "unused"]
    with pytest.raises(ValueError, match=message):
        args_to_config(parse_args([*base, *arguments]))


def test_unified_runner_evaluator_and_aggregators(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_root = tmp_path / "processed"
    results_root = tmp_path / "results"
    _write_processed_dataset(processed_root)

    run_dirs = [
        _run_train(repo_root, processed_root, results_root, "gru", "gru_unified"),
        _run_train(repo_root, processed_root, results_root, "kineroute_nvp", "kine_unified"),
        _run_train(repo_root, processed_root, results_root, "dead_reckoning", "dead_unified"),
    ]

    for run_dir in run_dirs:
        payload = json.loads((run_dir / "_results.json").read_text())
        assert payload["status"] == "completed"
        assert payload["model_parameter_count"] >= 0
        assert payload["model_metadata"]["name"] in {"gru", "kineroute_nvp", "dead_reckoning"}
        assert payload["seed"] == 42
        assert payload["best_epoch"] is not None
        assert payload["runtime_seconds"] >= 0.0
        assert (run_dir / "resolved_config.yaml").exists()
        assert (run_dir / "checkpoints" / "best.pt").exists()
        assert (run_dir / "checkpoints" / "final.pt").exists()
        assert (run_dir / "evaluation" / "metrics.json").exists()

    eval_output = tmp_path / "eval"
    subprocess.run(
        [sys.executable, "scripts/evaluate.py", str(run_dirs[0]), "--output-dir", str(eval_output), "--device", "cpu"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (eval_output / "metrics.json").exists()
    assert (eval_output / "provenance.json").exists()
    assert "ade" in json.loads((eval_output / "metrics.json").read_text())
    assert len(json.loads((eval_output / "provenance.json").read_text())["checkpoint_sha256"]) == 64

    aggregate_csv = tmp_path / "aggregate.csv"
    subprocess.run(
        [sys.executable, "scripts/aggregate_results.py", "--results-root", str(results_root), "--output", str(aggregate_csv)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    with aggregate_csv.open() as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert {row["model"] for row in aggregate_rows} == {"gru", "kineroute_nvp", "dead_reckoning"}

    multiseed_csv = tmp_path / "multiseed.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_multiseed.py",
            "--results-root",
            str(results_root),
            "--output",
            str(multiseed_csv),
            "--expected-seeds",
            "42",
            "--allow-unfrozen-data",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    with multiseed_csv.open() as handle:
        multiseed_rows = list(csv.DictReader(handle))
    assert len(multiseed_rows) == 3


def test_gru_raw_and_displacement_representations_differ(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    _write_processed_dataset(processed_root)
    dataset = ProcessedTrajectoryDataset(processed_root, "train", 4)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    raw = encode_with_history_torch(batch["history"], batch["history"], "raw")
    displacement = encode_with_history_torch(batch["history"], batch["history"], "heading_aligned_displacement")

    assert not torch.allclose(raw, displacement)


def test_heading_aligned_displacement_reaches_ordinary_forecaster(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    _write_processed_dataset(processed_root)
    dataset = ProcessedTrajectoryDataset(processed_root, "train", 4)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    runner = ExperimentRunner(_config(tmp_path, processed_root, "gru", "heading_aligned_displacement"), "unit", Path(__file__).resolve().parents[1], "test")
    model = CaptureForecaster()

    runner._predict_positions(model, batch, train=True, position_norm=None, chart_norm=None, chart_type="position")

    assert model.seen is not None
    expected = encode_with_history_torch(batch["history"], batch["history"], "heading_aligned_displacement")
    torch.testing.assert_close(model.seen, expected)


def test_kineroute_reverse_loss_contributes_to_objective(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    _write_processed_dataset(processed_root)
    config = _config(tmp_path, processed_root, "kineroute_nvp", "raw")
    config["loss"]["lambda_reverse"] = 3.0
    config["loss"]["lambda_position_mse"] = 0.4
    config["loss"]["lambda_fde"] = 0.2
    runner = ExperimentRunner(config, "unit", Path(__file__).resolve().parents[1], "test")
    predicted = torch.ones(2, 30, 2)
    target = torch.zeros(2, 30, 2)
    history = torch.zeros(2, 30, 2)
    reverse_ade = torch.tensor(2.0)

    objective, metrics = runner._loss(predicted, target, history, {"max_acceleration": 1e9, "max_turn_rate": 1e9}, None, None, reverse_ade)
    pieces = trajectory_loss(
        predicted_future=predicted,
        target_future=target,
        history_positions=history,
        dt_seconds=config["data"]["dt_seconds"],
        lambda_velocity=config["loss"]["lambda_velocity"],
        lambda_acceleration=config["loss"]["lambda_acceleration"],
        lambda_turn=config["loss"]["lambda_turn"],
        lambda_feasibility=config["loss"]["lambda_feasibility"],
        max_acceleration=1e9,
        max_turn_rate=1e9,
    )
    expected = pieces["total"] + 3.0 * reverse_ade + 0.4 * pieces["position_mse"] + 0.2 * pieces["fde"]

    torch.testing.assert_close(objective, expected)
    assert metrics["total"] == expected.item()


def test_dead_reckoning_bypasses_normalization(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    _write_processed_dataset(processed_root)
    dataset = ProcessedTrajectoryDataset(processed_root, "train", 4)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    config = _config(tmp_path, processed_root, "dead_reckoning", "heading_aligned_displacement")
    config["loss"]["normalize"] = True
    runner = ExperimentRunner(config, "unit", Path(__file__).resolve().parents[1], "test")
    model = DeadReckoningBaseline()
    fake_norm = (
        torch.full((1, 1, 2), 100.0),
        torch.full((1, 1, 2), 7.0),
        torch.full((1, 1, 2), -50.0),
        torch.full((1, 1, 2), 3.0),
    )

    pred_with_norm, _ = runner._predict_positions(model, batch, train=False, position_norm=fake_norm, chart_norm=None, chart_type="position")
    pred_without_norm, _ = runner._predict_positions(model, batch, train=False, position_norm=None, chart_norm=None, chart_type="position")

    torch.testing.assert_close(pred_with_norm, pred_without_norm)


def test_final_checkpoint_uses_actual_last_epoch_after_early_stopping(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_root = tmp_path / "processed"
    results_root = tmp_path / "results"
    _write_processed_dataset(processed_root)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--experiment-name",
            "early_stop",
            "--run-name",
            "gru_early_stop",
            "--processed-root",
            str(processed_root),
            "--results-root",
            str(results_root),
            "--model-name",
            "gru",
            "--hidden-dim",
            "8",
            "--num-layers",
            "1",
            "--epochs",
            "5",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--checkpoint-every-epochs",
            "1",
            "--early-stopping-patience",
            "0",
            "--learning-rate",
            "1e-30",
            "--device",
            "cpu",
            "--preprocessing-mode",
            "raw",
            "--loss-name",
            "mse",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(completed.stdout.strip().splitlines()[-1])
    payload = json.loads((run_dir / "_results.json").read_text())
    checkpoint = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)

    assert payload["last_completed_epoch"] == 2
    assert checkpoint["epoch"] == payload["last_completed_epoch"]


def test_validation_only_search_does_not_open_or_report_test_split(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_root = tmp_path / "processed"
    results_root = tmp_path / "results"
    _write_processed_dataset(processed_root)
    (processed_root / "test.npz").unlink()

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--experiment-name",
            "validation_only",
            "--run-name",
            "gru_search",
            "--processed-root",
            str(processed_root),
            "--results-root",
            str(results_root),
            "--model-name",
            "gru",
            "--hidden-dim",
            "8",
            "--num-layers",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--evaluate-test",
            "false",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = Path(completed.stdout.strip().splitlines()[-1])
    payload = json.loads((run_dir / "_results.json").read_text())

    assert payload["test_evaluated"] is False
    assert payload["test_ade"] is None
    assert payload["test_fde"] is None
    assert payload["evaluation_split"] == "validation"
    assert (run_dir / "evaluation" / "validation_metrics.json").exists()
    assert not (run_dir / "evaluation" / "metrics.json").exists()
