from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kineroute_nvp.data.preprocess import prepare_envship_dataset
from kineroute_nvp.utils.config import args_to_config, parse_args


def _resolve(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = args_to_config(args)
    data = config["data"]
    processed_root = _resolve(data["processed_root"])
    assert processed_root is not None
    prepare_envship_dataset(
        raw_root=_resolve(data.get("raw_root")),
        raw_roots=[path for value in data.get("raw_roots", []) if (path := _resolve(value)) is not None],
        processed_root=processed_root,
        dt_seconds=data["dt_seconds"],
        feasibility_quantile=config["loss"]["feasibility_quantile"],
        align_to_last_heading=data["align_to_last_heading"],
        chart_type=data["chart_type"],
        splits=(data["train_split"], data["val_split"], data["test_split"]),
        include_ship_classes=data.get("include_ship_classes"),
        quality_tiers=data.get("quality_tiers"),
        split_manifest_path=_resolve(data.get("split_manifest")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
