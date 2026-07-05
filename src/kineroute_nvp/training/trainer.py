from __future__ import annotations

import csv
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from kineroute_nvp.data.dataset import ProcessedTrajectoryDataset
from kineroute_nvp.data.preprocess import PrepareSummary, prepare_envship_dataset
from kineroute_nvp.evaluation.metrics import summarize_predictions
from kineroute_nvp.evaluation.plots import plot_loss_curves, plot_trajectory_examples
from kineroute_nvp.geometry.kinematics import decode_chart_torch
from kineroute_nvp.losses.trajectory_physics import ade, trajectory_loss
from kineroute_nvp.models.kineroute_nvp import KineRouteNVP
from kineroute_nvp.utils.io import append_jsonl, atomic_write_json, environment_snapshot, get_git_commit, write_json, write_yaml
from kineroute_nvp.utils.seed import set_seed


def _device_from_config(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


class ExperimentRunner:
    def __init__(self, config: dict, experiment_name: str, repo_root: Path, command: str) -> None:
        self.config = config
        self.experiment_name = experiment_name
        self.repo_root = repo_root
        self.command = command
        self.run_name = config["run_name"]
        self.results_root = repo_root / config["output"]["results_root"]
        self.device = _device_from_config(config["training"]["device"])
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = self.results_root / f"{self.timestamp}__{experiment_name}__{self.run_name}"
        self.result_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint_dir = self.result_dir / "checkpoints"
        self.log_dir = self.result_dir / "logs"
        self.figure_dir = self.result_dir / "figures"
        self.evaluation_dir = self.result_dir / "evaluation"
        for path in [self.checkpoint_dir, self.log_dir, self.figure_dir, self.evaluation_dir]:
            path.mkdir(parents=True, exist_ok=True)
        self.results_path = self.result_dir / "_results.json"
        self.metrics_jsonl = self.log_dir / "metrics.jsonl"
        self.logger = logging.getLogger(f"kineroute_nvp.{self.timestamp}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(self.log_dir / "train.log"))
        self.logger.addHandler(logging.StreamHandler())
        self._write_static_files()

    def _write_static_files(self) -> None:
        write_yaml(self.result_dir / "resolved_config.yaml", self.config)
        (self.result_dir / "command.txt").write_text(self.command + "\n")
        write_json(self.result_dir / "environment.json", environment_snapshot())

    def _update_results(self, payload: dict) -> None:
        atomic_write_json(self.results_path, payload)

    def _save_checkpoint(self, path: Path, model, optimizer, scheduler, epoch: int, best_val_ade: float) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "best_val_ade": best_val_ade,
                "rng_state": _rng_state(),
                "resolved_config": self.config,
            },
            path,
        )

    def _build_loaders(self, processed_root: Path) -> Tuple[DataLoader, DataLoader, DataLoader]:
        data_cfg = self.config["data"]
        train_dataset = ProcessedTrajectoryDataset(processed_root, data_cfg["train_split"], data_cfg.get("max_train_samples"))
        val_dataset = ProcessedTrajectoryDataset(processed_root, data_cfg["val_split"], data_cfg.get("max_val_samples"))
        test_dataset = ProcessedTrajectoryDataset(processed_root, data_cfg["test_split"], data_cfg.get("max_test_samples"))
        loader_kwargs = {
            "batch_size": self.config["training"]["batch_size"],
            "num_workers": self.config["training"]["num_workers"],
            "pin_memory": self.device.type == "cuda",
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        eval_loader_kwargs = dict(loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **eval_loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **eval_loader_kwargs)
        return train_loader, val_loader, test_loader

    def _denormalize_chart(self, normalized_chart: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return normalized_chart * std + mean

    def _run_epoch(
        self,
        model,
        loader,
        chart_type,
        history_chart_mean,
        history_chart_std,
        future_chart_mean,
        future_chart_std,
        optimizer,
        thresholds,
        train: bool,
    ) -> Dict[str, float]:
        mode = torch.enable_grad() if train else torch.no_grad()
        metrics = {key: 0.0 for key in ["total", "ade", "fde", "position_mse", "reverse_ade", "velocity", "acceleration", "turn", "feasibility"]}
        count = 0
        with mode:
            for batch in loader:
                history = batch["history"].to(self.device)
                future = batch["future"].to(self.device)
                history_chart = batch["history_chart"].to(self.device)
                future_chart = batch["future_chart"].to(self.device)
                predicted_chart_norm = model(history_chart)
                predicted_chart = self._denormalize_chart(predicted_chart_norm, future_chart_mean, future_chart_std)
                predicted_future = decode_chart_torch(predicted_chart, chart_type)
                reverse_history_chart = self._denormalize_chart(model.inverse(future_chart), history_chart_mean, history_chart_std)
                reverse_history = decode_chart_torch(reverse_history_chart, chart_type)
                loss_dict = trajectory_loss(
                    predicted_future=predicted_future,
                    target_future=future,
                    history_positions=history,
                    dt_seconds=self.config["data"]["dt_seconds"],
                    lambda_velocity=self.config["loss"]["lambda_velocity"],
                    lambda_acceleration=self.config["loss"]["lambda_acceleration"],
                    lambda_turn=self.config["loss"]["lambda_turn"],
                    lambda_feasibility=self.config["loss"]["lambda_feasibility"],
                    max_acceleration=thresholds["max_acceleration"],
                    max_turn_rate=thresholds["max_turn_rate"],
                )
                reverse_ade = ade(reverse_history, history)
                total_loss = (
                    loss_dict["total"]
                    + self.config["loss"]["lambda_reverse"] * reverse_ade
                    + self.config["loss"]["lambda_position_mse"] * loss_dict["position_mse"]
                    + self.config["loss"]["lambda_fde"] * loss_dict["fde"]
                )
                if train:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                batch_size = history.shape[0]
                count += batch_size
                metrics["total"] += float(total_loss.item()) * batch_size
                metrics["ade"] += float(loss_dict["ade"].item()) * batch_size
                metrics["fde"] += float(loss_dict["fde"].item()) * batch_size
                metrics["position_mse"] += float(loss_dict["position_mse"].item()) * batch_size
                metrics["reverse_ade"] += float(reverse_ade.item()) * batch_size
                metrics["velocity"] += float(loss_dict["velocity"].item()) * batch_size
                metrics["acceleration"] += float(loss_dict["acceleration"].item()) * batch_size
                metrics["turn"] += float(loss_dict["turn"].item()) * batch_size
                metrics["feasibility"] += float(loss_dict["feasibility"].item()) * batch_size
        return {key: value / max(count, 1) for key, value in metrics.items()}

    def _evaluate_split(self, model, loader, chart_type, history_chart_mean, history_chart_std, future_chart_mean, future_chart_std, thresholds) -> Tuple[Dict[str, float], dict]:
        history_batches = []
        future_batches = []
        predicted_batches = []
        reverse_batches = []
        inversion_error = 0.0
        with torch.no_grad():
            for batch in loader:
                history = batch["history"].to(self.device)
                future = batch["future"].to(self.device)
                history_chart = batch["history_chart"].to(self.device)
                future_chart = batch["future_chart"].to(self.device)
                predicted_chart_norm = model(history_chart)
                predicted_chart = self._denormalize_chart(predicted_chart_norm, future_chart_mean, future_chart_std)
                predicted_future = decode_chart_torch(predicted_chart, chart_type)
                reverse_history = decode_chart_torch(
                    self._denormalize_chart(model.inverse(future_chart), history_chart_mean, history_chart_std),
                    chart_type,
                )
                inversion_error = max(inversion_error, float(torch.max(torch.abs(model.inverse(model(history_chart)) - history_chart)).item()))
                history_batches.append(history.cpu())
                future_batches.append(future.cpu())
                predicted_batches.append(predicted_future.cpu())
                reverse_batches.append(reverse_history.cpu())
        history = torch.cat(history_batches)
        future = torch.cat(future_batches)
        predicted = torch.cat(predicted_batches)
        reverse_history = torch.cat(reverse_batches)
        summary = summarize_predictions(
            history=history,
            predicted_future=predicted,
            target_future=future,
            reverse_history=reverse_history,
            dt_seconds=self.config["data"]["dt_seconds"],
            max_acceleration=thresholds["max_acceleration"],
            max_turn_rate=thresholds["max_turn_rate"],
        )
        summary["invertibility_max_abs"] = inversion_error
        return summary, {
            "history": history.numpy(),
            "future": future.numpy(),
            "predicted": predicted.numpy(),
            "reverse_history": reverse_history.numpy(),
        }

    def run(self) -> Path:
        config = self.config
        results_payload = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "git_commit": get_git_commit(self.repo_root),
            "exact_command": self.command,
            "resolved_configuration": config,
            "dataset_paths": {
                "raw_root": config["data"]["raw_root"],
                "raw_roots": config["data"].get("raw_roots"),
                "processed_root": config["data"]["processed_root"],
            },
            "status": "running",
        }
        self._update_results(results_payload)
        try:
            set_seed(config["seed"])
            processed_root = self.repo_root / config["data"]["processed_root"]
            metadata_path = processed_root / "metadata.json"
            if not metadata_path.exists():
                if not config["data"]["prepare_if_missing"]:
                    raise FileNotFoundError("Processed data missing and prepare_if_missing is false.")
                summary = prepare_envship_dataset(
                    raw_root=self.repo_root / config["data"]["raw_root"] if config["data"].get("raw_root") else None,
                    raw_roots=[self.repo_root / root for root in config["data"].get("raw_roots", [])],
                    processed_root=processed_root,
                    dt_seconds=config["data"]["dt_seconds"],
                    feasibility_quantile=config["loss"]["feasibility_quantile"],
                    align_to_last_heading=config["data"].get("align_to_last_heading", False),
                    chart_type=config["data"].get("chart_type", "polar"),
                    splits=(config["data"]["train_split"], config["data"]["val_split"], config["data"]["test_split"]),
                )
            else:
                metadata = json.loads(metadata_path.read_text())
                summary = PrepareSummary(
                    processed_root=processed_root,
                    metadata_path=metadata_path,
                    normalization_path=processed_root / "normalization.json",
                    thresholds_path=processed_root / "feasibility_thresholds.json",
                    split_sizes=metadata["split_sizes"],
                )
            metadata = json.loads(summary.metadata_path.read_text())
            chart_type = metadata.get("chart_type", "polar")
            normalization = json.loads(summary.normalization_path.read_text())
            thresholds = json.loads(summary.thresholds_path.read_text())
            if "history_chart_mean" in normalization:
                history_chart_mean = torch.tensor(normalization["history_chart_mean"], dtype=torch.float32, device=self.device)
                history_chart_std = torch.tensor(normalization["history_chart_std"], dtype=torch.float32, device=self.device)
                future_chart_mean = torch.tensor(normalization["future_chart_mean"], dtype=torch.float32, device=self.device)
                future_chart_std = torch.tensor(normalization["future_chart_std"], dtype=torch.float32, device=self.device)
            else:
                shared_mean = torch.tensor(normalization["chart_mean"], dtype=torch.float32, device=self.device)
                shared_std = torch.tensor(normalization["chart_std"], dtype=torch.float32, device=self.device)
                history_chart_mean = shared_mean
                history_chart_std = shared_std
                future_chart_mean = shared_mean
                future_chart_std = shared_std
            train_loader, val_loader, test_loader = self._build_loaders(processed_root)
            model = KineRouteNVP(
                num_routed_blocks=config["model"]["num_routed_blocks"],
                hidden_dim=config["model"]["hidden_dim"],
                hidden_layers=config["model"]["hidden_layers"],
                scale_bound=config["model"]["scale_bound"],
            ).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
            scheduler = ReduceLROnPlateau(optimizer, factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"])
            best_val_ade = float("inf")
            best_epoch = 0
            epochs_without_improvement = 0
            for epoch in range(1, config["training"]["epochs"] + 1):
                start = time.time()
                model.train()
                train_metrics = self._run_epoch(
                    model,
                    train_loader,
                    chart_type,
                    history_chart_mean,
                    history_chart_std,
                    future_chart_mean,
                    future_chart_std,
                    optimizer,
                    thresholds,
                    train=True,
                )
                model.eval()
                val_metrics = self._run_epoch(
                    model,
                    val_loader,
                    chart_type,
                    history_chart_mean,
                    history_chart_std,
                    future_chart_mean,
                    future_chart_std,
                    optimizer,
                    thresholds,
                    train=False,
                )
                scheduler.step(val_metrics["ade"])
                record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "seconds": time.time() - start}
                append_jsonl(self.metrics_jsonl, record)
                self.logger.info("epoch=%s train_total=%.4f val_ade=%.4f", epoch, train_metrics["total"], val_metrics["ade"])
                if val_metrics["ade"] < best_val_ade:
                    best_val_ade = val_metrics["ade"]
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    self._save_checkpoint(self.checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_ade)
                else:
                    epochs_without_improvement += 1
                if epoch % config["training"]["checkpoint_every_epochs"] == 0:
                    self._save_checkpoint(self.checkpoint_dir / f"epoch_{epoch:04d}.pt", model, optimizer, scheduler, epoch, best_val_ade)
                if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
                    self.logger.info("Early stopping at epoch %s", epoch)
                    break

            final_epoch = epoch
            self._save_checkpoint(self.checkpoint_dir / "final.pt", model, optimizer, scheduler, final_epoch, best_val_ade)
            best_checkpoint = torch.load(self.checkpoint_dir / "best.pt", map_location=self.device, weights_only=False)
            model.load_state_dict(best_checkpoint["model_state"])
            model.eval()
            val_summary, _ = self._evaluate_split(
                model,
                val_loader,
                chart_type,
                history_chart_mean,
                history_chart_std,
                future_chart_mean,
                future_chart_std,
                thresholds,
            )
            test_summary, test_outputs = self._evaluate_split(
                model,
                test_loader,
                chart_type,
                history_chart_mean,
                history_chart_std,
                future_chart_mean,
                future_chart_std,
                thresholds,
            )
            write_json(self.evaluation_dir / "metrics.json", test_summary)
            with (self.evaluation_dir / "metrics.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["metric", "value"])
                for key, value in test_summary.items():
                    writer.writerow([key, value])
            write_json(
                self.evaluation_dir / "test_predictions_summary.json",
                {
                    "num_samples": int(test_outputs["history"].shape[0]),
                    "example_sample_count": min(config["evaluation"]["save_prediction_examples"], int(test_outputs["history"].shape[0])),
                },
            )
            plot_loss_curves(self.metrics_jsonl, self.figure_dir / "loss_curves.png")
            plot_trajectory_examples(
                test_outputs["history"],
                test_outputs["future"],
                test_outputs["predicted"],
                self.figure_dir / "trajectory_examples.png",
                max_examples=config["evaluation"]["save_prediction_examples"],
            )
            final_payload = {
                **results_payload,
                "dataset_split_sizes": summary.split_sizes,
                "train_derived_normalization_statistics_path": str(summary.normalization_path),
                "train_derived_feasibility_thresholds": thresholds,
                "best_epoch": best_epoch,
                "best_validation_ade": val_summary["ade"],
                "best_validation_fde": val_summary["fde"],
                "final_validation_ade": val_summary["ade"],
                "final_validation_fde": val_summary["fde"],
                "test_ade": test_summary["ade"],
                "test_fde": test_summary["fde"],
                "reverse_ade": test_summary["reverse_ade"],
                "reverse_fde": test_summary["reverse_fde"],
                "velocity_error": test_summary["velocity_error"],
                "acceleration_error": test_summary["acceleration_error"],
                "turn_rate_error": test_summary["turn_rate_error"],
                "infeasible_acceleration_percentage": test_summary["infeasible_acceleration_pct"],
                "infeasible_turn_rate_percentage": test_summary["infeasible_turn_rate_pct"],
                "invertibility_max_abs": test_summary["invertibility_max_abs"],
                "checkpoint_paths": {
                    "best": str(self.checkpoint_dir / "best.pt"),
                    "final": str(self.checkpoint_dir / "final.pt"),
                    "periodic": sorted(str(path) for path in self.checkpoint_dir.glob("epoch_*.pt")),
                },
                "status": "completed",
            }
            self._update_results(final_payload)
            return self.result_dir
        except Exception as exc:
            failed_payload = {**results_payload, "status": "failed", "exception": repr(exc)}
            self._update_results(failed_payload)
            raise
