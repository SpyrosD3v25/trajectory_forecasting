from __future__ import annotations

import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kineroute_nvp.training.trainer import ExperimentRunner
from kineroute_nvp.utils.config import args_to_config, parse_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = args_to_config(args)
    runner = ExperimentRunner(
        config=config,
        experiment_name=args.experiment_name,
        repo_root=REPO_ROOT,
        command="python scripts/train.py " + " ".join(shlex.quote(arg) for arg in (argv or sys.argv[1:])),
    )
    result_dir = runner.run()
    print(result_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
