from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kineroute_nvp.data.preprocess import prepare_envship_dataset
from kineroute_nvp.utils.config import parse_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = {
        "data": {
            "raw_root": args.data_root,
            "processed_root": args.processed_root,
            "dt_seconds": args.dt_seconds,
        },
        "loss": {
            "feasibility_quantile": args.feasibility_quantile,
        },
    }
    prepare_envship_dataset(
        raw_root=REPO_ROOT / config["data"]["raw_root"],
        processed_root=REPO_ROOT / config["data"]["processed_root"],
        dt_seconds=config["data"]["dt_seconds"],
        feasibility_quantile=config["loss"]["feasibility_quantile"],
        splits=(args.train_split, args.val_split, args.test_split),
    )
    return 0
