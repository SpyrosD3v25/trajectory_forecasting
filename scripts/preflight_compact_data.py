from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kineroute_nvp.data.split_manifest import validate_processed_dataset
from kineroute_nvp.utils.config import args_to_config, parse_args as parse_experiment_args


CANONICAL_CONFIGS = (
    "experiments/baselines/gru_coords_paper.yaml",
    "experiments/baselines/kine_real_nvp_paper.yaml",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the two compact processed roots against the immutable split manifest.")
    parser.add_argument("--repair", action="store_true", help="Regenerate an incompatible processed root from its frozen raw sources.")
    return parser.parse_args(argv)


def _resolved(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _validate(config_path: str) -> tuple[Path, dict]:
    config = args_to_config(parse_experiment_args(["--config", config_path]))
    processed_root = _resolved(config["data"]["processed_root"])
    manifest_path = _resolved(config["data"]["split_manifest"])
    manifest = validate_processed_dataset(processed_root, manifest_path, config)
    return processed_root, manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for config_path in CANONICAL_CONFIGS:
        try:
            processed_root, manifest = _validate(config_path)
            status = "valid"
        except (FileNotFoundError, KeyError, ValueError) as exc:
            if not args.repair:
                raise SystemExit(f"{config_path}: incompatible ({exc}); rerun with --repair") from exc
            subprocess.run(
                [sys.executable, "scripts/prepare_data.py", "--config", config_path],
                cwd=REPO_ROOT,
                check=True,
            )
            processed_root, manifest = _validate(config_path)
            status = "regenerated"
        print(f"{status}: {processed_root} counts={manifest['split_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
