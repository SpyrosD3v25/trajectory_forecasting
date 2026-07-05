from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ALIASES = {
    "dma_clean": "paper_dma_clean_ship_core_lite_v1",
    "noaa_clean": "paper_noaa_clean_ship_core_lite_v1",
    "dma_lite": "paper_dma_ship_core_lite",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a downloaded EnvShip paper subset.")
    parser.add_argument("--dataset", choices=sorted(ALIASES), default="dma_clean")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path("data/envship") / ALIASES[args.dataset]
    path = root / args.split / "part-000.csv.gz"
    summary_path = root / "summary.json"

    df = pd.read_csv(path)
    row = df.iloc[0]

    history = np.column_stack([json.loads(row["hist_x_json"]), json.loads(row["hist_y_json"])])
    future = np.column_stack([json.loads(row["fut_x_json"]), json.loads(row["fut_y_json"])])

    print(f"dataset_root={root}")
    print(f"split={args.split}")
    print(f"history_shape={history.shape}")
    print(f"future_shape={future.shape}")
    print(f"num_rows={len(df)}")
    print(f"summary_exists={summary_path.exists()}")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
