import json
import subprocess
import sys
from pathlib import Path


def _write_result(root: Path, name: str, seed: int, offset: float) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    payload = {
        "run_name": name,
        "seed": seed,
        "dataset_paths": {"raw_roots": ["data/envship/example"]},
        "dataset_split_sizes": {"train": 10, "val": 4, "test": 3},
        "resolved_configuration": {"seed": seed, "data": {"val_split": "val"}},
        "history": [
            {
                "epoch": epoch,
                "val": {"ade": offset + epoch, "fde": offset + 2 * epoch},
            }
            for epoch in range(1, 4)
        ],
    }
    (run_dir / "_results.json").write_text(json.dumps(payload))
    return run_dir


def test_plot_baseline_curves_accepts_explicit_folder_names(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = tmp_path / "results"
    _write_result(results_root, "gru_coords_paper", seed=42, offset=1.0)
    _write_result(results_root, "lstm_coords_paper", seed=42, offset=2.0)
    output_path = tmp_path / "combined.png"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/plot_baseline_curves.py",
            "gru_coords_paper",
            "lstm_coords_paper",
            "--results-root",
            str(results_root),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Seeds: [42]" in completed.stdout
    assert "Validation sample counts: [4]" in completed.stdout
    assert output_path.read_bytes().startswith(b"\x89PNG")
