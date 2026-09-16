import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from kineroute_nvp.training.artifacts import RunArtifacts


def test_training_plot_backend_is_headless() -> None:
    from kineroute_nvp.evaluation import plots

    assert "agg" in plots.matplotlib.get_backend().lower()


def _write_processed_split(root: Path, split: str, samples: int) -> None:
    history = np.zeros((samples, 30, 2), dtype=np.float32)
    future = np.zeros((samples, 30, 2), dtype=np.float32)
    for index in range(samples):
        slope = 0.05 * (index + 1)
        history[index, :, 0] = np.arange(30, dtype=np.float32) * slope
        future[index, :, 0] = np.arange(30, 60, dtype=np.float32) * slope
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
    sizes = {"train": 12, "val": 6, "test": 6}
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


def test_run_artifacts_exist_before_first_epoch(tmp_path: Path) -> None:
    artifacts = RunArtifacts(
        repo_root=tmp_path,
        results_root="results",
        experiment_name="smoke",
        run_name="artifact_contract",
        config={"training": {"checkpoint_every_epochs": 10}},
        command="python train.py",
        timestamp="20260101_000000",
    )

    payload = json.loads(artifacts.results_path.read_text())
    assert payload["status"] == "running"
    assert payload["history"] == []
    assert artifacts.val_ade_plot.read_bytes().startswith(b"\x89PNG")
    assert artifacts.val_fde_plot.read_bytes().startswith(b"\x89PNG")


def test_periodic_checkpoint_contains_resumable_state(tmp_path: Path) -> None:
    artifacts = RunArtifacts(
        repo_root=tmp_path,
        results_root="results",
        experiment_name="smoke",
        run_name="checkpoint_contract",
        config={"training": {"checkpoint_every_epochs": 10}},
        command="python train.py",
        timestamp="20260101_000001",
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    assert artifacts.save_periodic_checkpoint(
        model=model, optimizer=optimizer, scheduler=scheduler, epoch=9, interval=10
    ) is None
    checkpoint_path = artifacts.save_periodic_checkpoint(
        model=model, optimizer=optimizer, scheduler=scheduler, epoch=10, interval=10
    )

    assert checkpoint_path == artifacts.checkpoint_dir / "epoch_0010.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 10
    assert checkpoint["optimizer_state"] is not None
    assert checkpoint["scheduler_state"] is not None
    assert checkpoint["rng_state"]["torch"] is not None


def test_baseline_training_uses_shared_artifact_contract(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_root = tmp_path / "processed"
    results_root = tmp_path / "results"
    _write_processed_dataset(processed_root)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline.py",
            "--experiment-name",
            "smoke",
            "--run-name",
            "baseline_artifact_contract",
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
            "10",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--checkpoint-every-epochs",
            "10",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    run_dir = next(results_root.glob("*__smoke__baseline_artifact_contract"))
    payload = json.loads((run_dir / "_results.json").read_text())
    assert payload["status"] == "completed"
    assert len(payload["runtime_source_identity"]["sha256"]) == 64
    assert "scripts/train.py" in payload["runtime_source_identity"]["files"]
    assert payload["last_completed_epoch"] == 10
    assert len(payload["history"]) == 10
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "checkpoints" / "epoch_0010.pt").exists()
    assert (run_dir / "checkpoints" / "final.pt").exists()
    assert (run_dir / "figures" / "val_ade_vs_epoch.png").exists()
    assert (run_dir / "figures" / "val_fde_vs_epoch.png").exists()
    assert (run_dir / "figures" / "trajectory_examples.png").exists()
    assert (run_dir / "evaluation" / "metrics.csv").exists()
