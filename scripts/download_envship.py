from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "mark000071/EnvShip-Bench_An_Environment-Enhanced_Benchmark_for_Short-Term_Vessel_Trajectory_Prediction"
OUT_DIR = Path("data/envship")

DATASETS = {
    "dma_clean": {
        "remote_root": "DMA/mini_benchmark/clean_ship_core_lite_v1",
        "alias_root": "paper_dma_clean_ship_core_lite_v1",
    },
    "noaa_clean": {
        "remote_root": "NOAA/mini_bench/clean_ship_core_lite_v1",
        "alias_root": "paper_noaa_clean_ship_core_lite_v1",
    },
    "dma_lite": {
        "remote_root": "DMA/mini_benchmark/ship_core_lite",
        "alias_root": "paper_dma_ship_core_lite",
    },
}

DOC_PATTERNS = ["README.md"]


def build_patterns(keys: list[str], include_context: bool) -> list[str]:
    patterns = list(DOC_PATTERNS)
    for key in keys:
        remote_root = DATASETS[key]["remote_root"]
        patterns.extend(
            [
                f"{remote_root}/train/**",
                f"{remote_root}/val/**",
                f"{remote_root}/test/**",
                f"{remote_root}/summary.json",
                f"{remote_root}/reports/**",
                f"{remote_root}/sample_ids/**",
                f"{remote_root}/README.md",
            ]
        )
        if include_context:
            patterns.extend(
                [
                    f"{remote_root}/environment_v1/**",
                    f"{remote_root}/environment_v2/**",
                    f"{remote_root}/social_env_v1/**",
                ]
            )
    return patterns


def create_alias(remote_root: str, alias_root: str) -> None:
    source_root = OUT_DIR / remote_root
    target_root = OUT_DIR / alias_root
    target_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_dir = target_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        source_file = source_root / split / "part-000.csv.gz"
        target_file = split_dir / "part-000.csv.gz"
        if target_file.exists() or target_file.is_symlink():
            target_file.unlink()
        target_file.symlink_to(source_file.resolve())
    summary_link = target_root / "summary.json"
    if summary_link.exists() or summary_link.is_symlink():
        summary_link.unlink()
    summary_link.symlink_to((source_root / "summary.json").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official EnvShip-Bench paper subsets from Hugging Face.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["dma_clean"],
        help="Which official paper subsets to download.",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Also download environment/social context packages for the selected clean subsets.",
    )
    return parser.parse_args()


