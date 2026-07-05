from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedTrajectoryDataset(Dataset):
    def __init__(self, processed_root: Path, split: str, max_samples: int | None = None) -> None:
        self.processed_root = Path(processed_root)
        npz = np.load(self.processed_root / f"{split}.npz", allow_pickle=True)
        metadata = json.loads((self.processed_root / "metadata.json").read_text())
        normalization = json.loads((self.processed_root / "normalization.json").read_text())
        limit = max_samples if max_samples is not None else len(npz["history"])
        self.sample_id = npz["sample_id"][:limit]
        self.history = npz["history"][:limit].astype(np.float32)
        self.future = npz["future"][:limit].astype(np.float32)
        if "history_chart_mean" in normalization:
            history_chart_mean = np.asarray(normalization["history_chart_mean"], dtype=np.float32)
            history_chart_std = np.asarray(normalization["history_chart_std"], dtype=np.float32)
            future_chart_mean = np.asarray(normalization["future_chart_mean"], dtype=np.float32)
            future_chart_std = np.asarray(normalization["future_chart_std"], dtype=np.float32)
        else:
            chart_mean = np.asarray(normalization["chart_mean"], dtype=np.float32)
            chart_std = np.asarray(normalization["chart_std"], dtype=np.float32)
            history_chart_mean = chart_mean
            history_chart_std = chart_std
            future_chart_mean = chart_mean
            future_chart_std = chart_std
        self.history_chart = ((npz["history_chart"][:limit].astype(np.float32) - history_chart_mean) / history_chart_std).astype(np.float32)
        self.future_chart = ((npz["future_chart"][:limit].astype(np.float32) - future_chart_mean) / future_chart_std).astype(np.float32)
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "sample_id": str(self.sample_id[index]),
            "history": torch.from_numpy(self.history[index]),
            "future": torch.from_numpy(self.future[index]),
            "history_chart": torch.from_numpy(self.history_chart[index]),
            "future_chart": torch.from_numpy(self.future_chart[index]),
        }
