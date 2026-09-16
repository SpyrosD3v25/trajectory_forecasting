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
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive when provided")
        metadata = json.loads((self.processed_root / "metadata.json").read_text())
        normalization = json.loads((self.processed_root / "normalization.json").read_text())
        with np.load(self.processed_root / f"{split}.npz", allow_pickle=True) as npz:
            limit = max_samples if max_samples is not None else len(npz["history"])
            self.sample_id = npz["sample_id"][:limit]
            self.history = npz["history"][:limit].astype(np.float32)
            self.future = npz["future"][:limit].astype(np.float32)
            history_chart = npz["history_chart"][:limit].astype(np.float32)
            future_chart = npz["future_chart"][:limit].astype(np.float32)
        if self.history.ndim != 3 or self.history.shape[1:] != (30, 2):
            raise ValueError(f"{split} history must have shape [N, 30, 2], got {self.history.shape}")
        if self.future.ndim != 3 or self.future.shape[1:] != (30, 2):
            raise ValueError(f"{split} future must have shape [N, 30, 2], got {self.future.shape}")
        if not (len(self.sample_id) == len(self.history) == len(self.future)):
            raise ValueError(f"{split} sample IDs, history, and future have inconsistent lengths")
        if int(metadata.get("history_steps", 30)) != 30 or int(metadata.get("future_steps", 30)) != 30:
            raise ValueError("Processed metadata violates the required 30 -> 30 protocol")
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
        statistics = (history_chart_mean, history_chart_std, future_chart_mean, future_chart_std)
        if any(values.shape != (60,) or not np.isfinite(values).all() for values in statistics):
            raise ValueError("Chart normalization statistics must be finite vectors of length 60")
        if np.any(history_chart_std <= 0.0) or np.any(future_chart_std <= 0.0):
            raise ValueError("Chart normalization standard deviations must be positive")
        if history_chart.shape != (len(self.history), 60) or future_chart.shape != (len(self.future), 60):
            raise ValueError(f"{split} chart arrays must have shape [N, 60]")
        self.history_chart = ((history_chart - history_chart_mean) / history_chart_std).astype(np.float32)
        self.future_chart = ((future_chart - future_chart_mean) / future_chart_std).astype(np.float32)
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
