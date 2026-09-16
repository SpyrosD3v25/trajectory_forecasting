import json
import shutil
import subprocess
from pathlib import Path


def test_smoke_pipeline_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_root = repo_root / "data/processed/test_smoke_envship"
    results_root = repo_root / "test_results"
    shutil.rmtree(processed_root, ignore_errors=True)
    shutil.rmtree(results_root, ignore_errors=True)
    command = [
        "python",
        "scripts/train.py",
        "--config",
        "experiments/smoke/cpu_smoke.yaml",
        "--run-name",
        "test_smoke_pipeline",
        "--processed-root",
        "data/processed/test_smoke_envship",
        "--results-root",
        "test_results",
    ]
    subprocess.run(command, cwd=repo_root, check=True)
    result_dirs = list(results_root.glob("*__smoke__test_smoke_pipeline"))
    assert len(result_dirs) == 1
    result_dir = result_dirs[0]
    payload = json.loads((result_dir / "_results.json").read_text())
    assert payload["status"] == "completed"
    assert (result_dir / "checkpoints" / "best.pt").exists()
    assert (result_dir / "checkpoints" / "final.pt").exists()
    assert (result_dir / "checkpoints" / "epoch_0001.pt").exists()
    assert (result_dir / "figures" / "val_ade_vs_epoch.png").exists()
    assert (result_dir / "figures" / "val_fde_vs_epoch.png").exists()
    assert (result_dir / "evaluation" / "metrics.csv").exists()
