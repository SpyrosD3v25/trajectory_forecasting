from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kineroute_nvp.data.split_manifest import create_split_manifest


DEFAULT_ROOTS = [
    REPO_ROOT / "data/envship/paper_dma_clean_ship_core_lite_v1",
    REPO_ROOT / "data/envship/paper_noaa_clean_ship_core_lite_v1",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze one compact EnvShip split for all models and training seeds.")
    parser.add_argument("--raw-roots", nargs="+", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/compact/split_manifest.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = create_split_manifest(
        raw_roots=args.raw_roots,
        repo_root=REPO_ROOT,
        dataset_id="envship_compact_dma_noaa_clean_ship_core_lite_v1",
        history_steps=30,
        future_steps=30,
        dt_seconds=20.0,
        coordinate_unit="metres",
        coordinate_frame="local planar anchor-relative frame; x=east, y=north",
        coordinate_unit_evidence={
            "path": "data/envship/DATA_CARD.md",
            "sha256": hashlib.sha256((REPO_ROOT / "data/envship/DATA_CARD.md").read_bytes()).hexdigest(),
            "statement": "All positions are in a local planar frame centred on the last history point; x=east, y=north, in metres.",
        },
        sampling_interval_evidence={
            "path": "data/envship/README.md",
            "sha256": hashlib.sha256((REPO_ROOT / "data/envship/README.md").read_bytes()).hexdigest(),
            "statement": "Sampling interval: 20 seconds; history length: 30 points; future length: 30 points.",
        },
        training_seeds=(0, 1, 2, 3, 4),
    )
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
