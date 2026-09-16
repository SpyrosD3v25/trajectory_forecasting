from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kineroute_nvp.evaluation.plots import plot_validation_curves
from kineroute_nvp.utils.io import append_jsonl, atomic_write_json, environment_snapshot, write_json, write_yaml


ARTIFACT_SCHEMA_VERSION = 2


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


class RunArtifacts:
    """Owns the durable, model-independent artifacts for one training run."""

    def __init__(
        self,
        *,
        repo_root: Path,
        results_root: str | Path,
        experiment_name: str,
        run_name: str,
        config: dict,
        command: str,
        initial_results: dict | None = None,
        timestamp: str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.config = config
        self.command = command
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(results_root)
        if not root.is_absolute():
            root = self.repo_root / root
        self.result_dir = root / f"{self.timestamp}__{experiment_name}__{run_name}"
        self.checkpoint_dir = self.result_dir / "checkpoints"
        self.log_dir = self.result_dir / "logs"
        self.figure_dir = self.result_dir / "figures"
        self.evaluation_dir = self.result_dir / "evaluation"
        self.results_path = self.result_dir / "_results.json"
        self.metrics_jsonl = self.log_dir / "metrics.jsonl"
        self.val_ade_plot = self.figure_dir / "val_ade_vs_epoch.png"
        self.val_fde_plot = self.figure_dir / "val_fde_vs_epoch.png"

        self.result_dir.mkdir(parents=True, exist_ok=False)
        for path in (self.checkpoint_dir, self.log_dir, self.figure_dir, self.evaluation_dir):
            path.mkdir(parents=True, exist_ok=True)

        write_yaml(self.result_dir / "resolved_config.yaml", config)
        (self.result_dir / "command.txt").write_text(command + "\n")
        write_json(self.result_dir / "environment.json", environment_snapshot())

        self.history: list[dict] = []
        self.results = {
            **(initial_results or {}),
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "timestamp": self.timestamp,
            "status": "running",
            "last_completed_epoch": 0,
            "history": self.history,
            "artifacts": {
                "metrics_jsonl": str(self.metrics_jsonl),
                "val_ade_plot": str(self.val_ade_plot),
                "val_fde_plot": str(self.val_fde_plot),
                "checkpoint_directory": str(self.checkpoint_dir),
            },
        }
        plot_validation_curves(self.history, self.figure_dir)
        self._flush_results()

    def create_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"kineroute_nvp.{self.experiment_name}.{self.timestamp}.{self.run_name}")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(logging.FileHandler(self.log_dir / "train.log"))
        logger.addHandler(logging.StreamHandler())
        return logger

    def _flush_results(self) -> None:
        atomic_write_json(self.results_path, self.results)

    def update(self, **values: Any) -> None:
        self.results.update(values)
        self._flush_results()

    def record_epoch(self, record: dict) -> None:
        epoch = int(record["epoch"])
        append_jsonl(self.metrics_jsonl, record)
        self.history.append(record)
        self.results["last_completed_epoch"] = epoch
        plot_validation_curves(self.history, self.figure_dir)
        self._flush_results()

    def save_checkpoint(
        self,
        filename: str,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any,
        epoch: int,
        selection: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = self.checkpoint_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_state": _cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": int(epoch),
            "selection": selection or {},
            "rng_state": _rng_state(),
            "resolved_config": self.config,
            "run_provenance": {
                key: self.results.get(key)
                for key in ("git_commit", "runtime_source_identity", "dataset_identity")
                if self.results.get(key) is not None
            },
            **(extra or {}),
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return path

    def save_periodic_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        interval: int,
        selection: dict[str, Any] | None = None,
    ) -> Path | None:
        if interval <= 0:
            raise ValueError("checkpoint interval must be positive")
        if epoch % interval != 0:
            return None
        return self.save_checkpoint(
            f"epoch_{epoch:04d}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            selection=selection,
        )

    def complete(self, **values: Any) -> None:
        plot_validation_curves(self.history, self.figure_dir)
        self.results.update(values)
        self.results["status"] = "completed"
        self._flush_results()

    def fail(self, exc: BaseException) -> None:
        plot_validation_curves(self.history, self.figure_dir)
        self.results.update(status="failed", exception=repr(exc))
        self._flush_results()
