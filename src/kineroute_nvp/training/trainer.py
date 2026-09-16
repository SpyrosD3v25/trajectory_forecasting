from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from kineroute_nvp.data.dataset import ProcessedTrajectoryDataset
from kineroute_nvp.data.preprocess import PrepareSummary, prepare_envship_dataset
from kineroute_nvp.data.split_manifest import load_split_manifest, sha256_file, validate_experiment_protocol, validate_processed_dataset
from kineroute_nvp.data.trajectory_preprocessing import decode_with_history_torch, encode_with_history_torch
from kineroute_nvp.evaluation.metrics import summarize_predictions
from kineroute_nvp.evaluation.plots import plot_loss_curves, plot_trajectory_examples
from kineroute_nvp.geometry.kinematics import decode_chart_torch
from kineroute_nvp.losses.trajectory_physics import ade, fde, mse, trajectory_loss
from kineroute_nvp.models.baselines import DeadReckoningBaseline, Seq2SeqBaseline
from kineroute_nvp.models.factory import build_model, model_metadata
from kineroute_nvp.training.artifacts import RunArtifacts
from kineroute_nvp.utils.io import get_git_commit, runtime_source_identity, write_json
from kineroute_nvp.utils.model import count_trainable_parameters
from kineroute_nvp.utils.seed import set_seed


CHART_MODELS = {"kineroute_nvp", "routed_realnvp", "realnvp", "non_invertible_routed"}


def _device_from_config(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_repo_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _default_thresholds() -> dict[str, float]:
    return {"max_acceleration": 1e9, "max_turn_rate": 1e9}


class ExperimentRunner:
    def __init__(self, config: dict, experiment_name: str, repo_root: Path, command: str) -> None:
        self.config = config
        self.experiment_name = experiment_name
        self.repo_root = repo_root
        self.command = command
        self.run_name = config["run_name"]
        self.device = _device_from_config(config["training"]["device"])
        self.artifacts = RunArtifacts(
            repo_root=repo_root,
            results_root=config["output"]["results_root"],
            experiment_name=experiment_name,
            run_name=self.run_name,
            config=config,
            command=command,
            initial_results={
                "git_commit": get_git_commit(repo_root),
                "runtime_source_identity": runtime_source_identity(repo_root),
                "exact_command": command,
                "resolved_configuration": config,
                "seed": config["seed"],
                "model": config["model"]["name"],
                "dataset_paths": {
                    "raw_root": config["data"].get("raw_root"),
                    "raw_roots": config["data"].get("raw_roots"),
                    "processed_root": config["data"]["processed_root"],
                },
            },
        )
        self.result_dir = self.artifacts.result_dir
        self.checkpoint_dir = self.artifacts.checkpoint_dir
        self.figure_dir = self.artifacts.figure_dir
        self.evaluation_dir = self.artifacts.evaluation_dir
        self.metrics_jsonl = self.artifacts.metrics_jsonl
        self.logger = self.artifacts.create_logger()

    @property
    def model_name(self) -> str:
        return self.config["model"]["name"]

    @property
    def uses_chart_model(self) -> bool:
        return self.model_name in CHART_MODELS

    def _prepare_data(self) -> PrepareSummary:
        data_cfg = self.config["data"]
        processed_root = _resolve_repo_path(self.repo_root, data_cfg["processed_root"])
        assert processed_root is not None
        metadata_path = processed_root / "metadata.json"
        manifest_path = _resolve_repo_path(self.repo_root, data_cfg.get("split_manifest"))
        if not metadata_path.exists():
            if not data_cfg["prepare_if_missing"]:
                raise FileNotFoundError("Processed data missing and prepare_if_missing is false.")
            summary = prepare_envship_dataset(
                raw_root=_resolve_repo_path(self.repo_root, data_cfg.get("raw_root")),
                raw_roots=[_resolve_repo_path(self.repo_root, root) for root in data_cfg.get("raw_roots", [])],
                processed_root=processed_root,
                dt_seconds=data_cfg["dt_seconds"],
                feasibility_quantile=self.config["loss"].get("feasibility_quantile", 0.995),
                align_to_last_heading=data_cfg.get("align_to_last_heading", False),
                chart_type=data_cfg.get("chart_type", "position"),
                splits=(data_cfg["train_split"], data_cfg["val_split"], data_cfg["test_split"]),
                include_ship_classes=data_cfg.get("include_ship_classes"),
                quality_tiers=data_cfg.get("quality_tiers"),
                split_manifest_path=manifest_path,
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
        manifest = validate_processed_dataset(processed_root, manifest_path, self.config) if manifest_path is not None else None
        validate_experiment_protocol(self.config, metadata, manifest)
        return summary

    def _build_loaders(self, processed_root: Path) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        data_cfg = self.config["data"]
        train_dataset = ProcessedTrajectoryDataset(processed_root, data_cfg["train_split"], data_cfg.get("max_train_samples"))
        val_dataset = ProcessedTrajectoryDataset(processed_root, data_cfg["val_split"], data_cfg.get("max_val_samples"))
        test_dataset = (
            ProcessedTrajectoryDataset(processed_root, data_cfg["test_split"], data_cfg.get("max_test_samples"))
            if self.config["evaluation"].get("evaluate_test", True)
            else None
        )
        loader_kwargs = {
            "batch_size": self.config["training"]["batch_size"],
            "num_workers": self.config["training"]["num_workers"],
            "pin_memory": self.device.type == "cuda",
        }
        generator = torch.Generator().manual_seed(self.config["seed"])
        return (
            DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs),
            DataLoader(val_dataset, shuffle=False, **loader_kwargs),
            DataLoader(test_dataset, shuffle=False, **loader_kwargs) if test_dataset is not None else None,
        )

    def _position_normalization(self, train_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.config["loss"].get("normalize", True) or self.uses_chart_model or self.model_name == "dead_reckoning":
            return None
        dataset = train_loader.dataset
        train_history_metric = torch.as_tensor(dataset.history, dtype=torch.float32, device=self.device)
        train_future_metric = torch.as_tensor(dataset.future, dtype=torch.float32, device=self.device)
        mode = self.config["data"]["preprocessing_mode"]
        train_history = encode_with_history_torch(train_history_metric, train_history_metric, mode)
        train_future = encode_with_history_torch(train_history_metric, train_future_metric, mode)
        # Normalize each trajectory feature independently.  This is the tensor
        # form of the per-dimension chart normalization used by invertible
        # models, avoiding a normalization confound in architecture ablations.
        history_mean = train_history.mean(dim=0, keepdim=True).to(self.device)
        history_std = train_history.std(dim=0, keepdim=True, correction=0).clamp_min(1e-6).to(self.device)
        future_mean = train_future.mean(dim=0, keepdim=True).to(self.device)
        future_std = train_future.std(dim=0, keepdim=True, correction=0).clamp_min(1e-6).to(self.device)
        return history_mean, history_std, future_mean, future_std

    def _chart_normalization(self, summary: PrepareSummary) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalization = json.loads(summary.normalization_path.read_text())
        if "history_chart_mean" in normalization:
            return (
                torch.tensor(normalization["history_chart_mean"], dtype=torch.float32, device=self.device),
                torch.tensor(normalization["history_chart_std"], dtype=torch.float32, device=self.device),
                torch.tensor(normalization["future_chart_mean"], dtype=torch.float32, device=self.device),
                torch.tensor(normalization["future_chart_std"], dtype=torch.float32, device=self.device),
            )
        shared_mean = torch.tensor(normalization["chart_mean"], dtype=torch.float32, device=self.device)
        shared_std = torch.tensor(normalization["chart_std"], dtype=torch.float32, device=self.device)
        return shared_mean, shared_std, shared_mean, shared_std

    def _predict_positions(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        *,
        train: bool,
        position_norm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        chart_norm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        chart_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        history = batch["history"].to(self.device)
        future = batch["future"].to(self.device)
        if self.uses_chart_model:
            assert chart_norm is not None
            _, _, future_chart_mean, future_chart_std = chart_norm
            predicted_chart_norm = model(batch["history_chart"].to(self.device))
            predicted_chart = predicted_chart_norm * future_chart_std + future_chart_mean
            return decode_chart_torch(predicted_chart, chart_type), predicted_chart_norm
        if isinstance(model, DeadReckoningBaseline):
            return model(history), None
        mode = self.config["data"]["preprocessing_mode"]
        model_history = encode_with_history_torch(history, history, mode)
        model_future = encode_with_history_torch(history, future, mode)
        if position_norm is None:
            if isinstance(model, Seq2SeqBaseline):
                predicted_encoded = model(model_history, model_future if train else None, self.config["loss"].get("teacher_forcing_ratio", 0.0) if train else 0.0)
            else:
                predicted_encoded = model(model_history)
            return decode_with_history_torch(history, predicted_encoded, mode), predicted_encoded
        history_mean, history_std, future_mean, future_std = position_norm
        norm_history = (model_history - history_mean) / history_std
        if isinstance(model, Seq2SeqBaseline):
            norm_future = (model_future - future_mean) / future_std
            pred_norm = model(norm_history, norm_future if train else None, self.config["loss"].get("teacher_forcing_ratio", 0.0) if train else 0.0)
        else:
            pred_norm = model(norm_history)
        predicted_encoded = pred_norm * future_std + future_mean
        return decode_with_history_torch(history, predicted_encoded, mode), pred_norm

    def _loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        history: torch.Tensor,
        thresholds: dict[str, float],
        predicted_norm: torch.Tensor | None,
        future_norm: torch.Tensor | None,
        reverse_ade: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_name = self.config["loss"]["name"]
        if loss_name == "mse":
            objective = mse(predicted_norm, future_norm) if predicted_norm is not None and future_norm is not None else mse(predicted, target)
        elif loss_name == "ade":
            objective = ade(predicted, target)
        elif loss_name == "kineroute":
            pieces = trajectory_loss(
                predicted_future=predicted,
                target_future=target,
                history_positions=history,
                dt_seconds=self.config["data"]["dt_seconds"],
                lambda_velocity=self.config["loss"].get("lambda_velocity", 0.0),
                lambda_acceleration=self.config["loss"].get("lambda_acceleration", 0.0),
                lambda_turn=self.config["loss"].get("lambda_turn", 0.0),
                lambda_feasibility=self.config["loss"].get("lambda_feasibility", 0.0),
                max_acceleration=thresholds["max_acceleration"],
                max_turn_rate=thresholds["max_turn_rate"],
            )
            reverse_component = torch.zeros((), dtype=predicted.dtype, device=predicted.device) if reverse_ade is None else reverse_ade
            objective = (
                pieces["total"]
                + self.config["loss"].get("lambda_reverse", 0.0) * reverse_component
                + self.config["loss"].get("lambda_position_mse", 0.0) * pieces["position_mse"]
                + self.config["loss"].get("lambda_fde", 0.0) * pieces["fde"]
            )
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")
        return objective, {
            "total": float(objective.detach().item()),
            "ade": float(ade(predicted, target).detach().item()),
            "fde": float(fde(predicted, target).detach().item()),
            "position_mse": float(mse(predicted, target).detach().item()),
        }

    def _run_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        thresholds: dict[str, float],
        position_norm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        chart_norm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        chart_type: str,
        train: bool,
    ) -> dict[str, float]:
        model.train(train)
        totals = {key: 0.0 for key in ("total", "ade", "fde", "position_mse", "reverse_ade")}
        count = 0
        context = torch.enable_grad() if train and optimizer is not None else torch.no_grad()
        with context:
            for batch in loader:
                history = batch["history"].to(self.device)
                future = batch["future"].to(self.device)
                predicted, predicted_norm = self._predict_positions(model, batch, train=train, position_norm=position_norm, chart_norm=chart_norm, chart_type=chart_type)
                future_norm = None
                if self.uses_chart_model and predicted_norm is not None:
                    future_norm = batch["future_chart"].to(self.device)
                elif position_norm is not None and predicted_norm is not None:
                    _, _, future_mean, future_std = position_norm
                    future_norm = (encode_with_history_torch(history, future, self.config["data"]["preprocessing_mode"]) - future_mean) / future_std
                reverse_ade = None
                if (
                    self.uses_chart_model
                    and hasattr(model, "inverse")
                    and chart_norm is not None
                    and (
                        self.config["loss"].get("lambda_reverse", 0.0) != 0.0
                        or (not train and self.config["evaluation"].get("evaluate_reverse_mapping", True))
                    )
                ):
                    history_chart_mean, history_chart_std, _, _ = chart_norm
                    reverse_chart = model.inverse(batch["future_chart"].to(self.device)) * history_chart_std + history_chart_mean
                    reverse_ade = ade(decode_chart_torch(reverse_chart, chart_type), history)
                objective, metrics = self._loss(predicted, future, history, thresholds, predicted_norm, future_norm, reverse_ade)
                metrics["reverse_ade"] = 0.0 if reverse_ade is None else float(reverse_ade.detach().item())
                if train and optimizer is not None:
                    optimizer.zero_grad()
                    objective.backward()
                    optimizer.step()
                batch_size = history.shape[0]
                count += batch_size
                for key, value in metrics.items():
                    totals[key] += value * batch_size
        return {key: value / max(count, 1) for key, value in totals.items()}

    def evaluate_split(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        thresholds: dict[str, float],
        position_norm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        chart_norm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        chart_type: str,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        model.eval()
        history_batches = []
        future_batches = []
        predicted_batches = []
        reverse_batches = []
        sample_ids: list[str] = []
        inversion_error = 0.0
        with torch.no_grad():
            for batch in loader:
                history = batch["history"].to(self.device)
                future = batch["future"].to(self.device)
                predicted, _ = self._predict_positions(model, batch, train=False, position_norm=position_norm, chart_norm=chart_norm, chart_type=chart_type)
                reverse_history = history
                if (
                    self.uses_chart_model
                    and hasattr(model, "inverse")
                    and chart_norm is not None
                    and self.config["evaluation"].get("evaluate_reverse_mapping", True)
                ):
                    history_chart_mean, history_chart_std, _, _ = chart_norm
                    history_chart = batch["history_chart"].to(self.device)
                    future_chart = batch["future_chart"].to(self.device)
                    reverse_chart = model.inverse(future_chart) * history_chart_std + history_chart_mean
                    reverse_history = decode_chart_torch(reverse_chart, chart_type)
                    inversion_error = max(inversion_error, float(torch.max(torch.abs(model.inverse(model(history_chart)) - history_chart)).item()))
                history_batches.append(history.cpu())
                future_batches.append(future.cpu())
                predicted_batches.append(predicted.cpu())
                reverse_batches.append(reverse_history.cpu())
                sample_ids.extend(batch["sample_id"])
        history_tensor = torch.cat(history_batches)
        future_tensor = torch.cat(future_batches)
        predicted_tensor = torch.cat(predicted_batches)
        reverse_tensor = torch.cat(reverse_batches)
        summary = summarize_predictions(
            history=history_tensor,
            predicted_future=predicted_tensor,
            target_future=future_tensor,
            reverse_history=reverse_tensor,
            dt_seconds=self.config["data"]["dt_seconds"],
            max_acceleration=thresholds["max_acceleration"],
            max_turn_rate=thresholds["max_turn_rate"],
        )
        if not (self.uses_chart_model and hasattr(model, "inverse")):
            summary["reverse_ade"] = 0.0
            summary["reverse_fde"] = 0.0
        summary["invertibility_max_abs"] = inversion_error
        return summary, {
            "history": history_tensor.numpy(),
            "future": future_tensor.numpy(),
            "predicted": predicted_tensor.numpy(),
            "reverse_history": reverse_tensor.numpy(),
            "sample_ids": sample_ids,
        }

    def _save_metrics(self, metrics: dict[str, float], split: str) -> tuple[Path, Path]:
        stem = "metrics" if split == "test" else f"{split}_metrics"
        json_path = self.evaluation_dir / f"{stem}.json"
        csv_path = self.evaluation_dir / f"{stem}.csv"
        write_json(json_path, metrics)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
        return json_path, csv_path

    def run(self) -> Path:
        config = self.config
        try:
            started_run = time.time()
            set_seed(config["seed"])
            summary = self._prepare_data()
            metadata = json.loads(summary.metadata_path.read_text())
            manifest_path = _resolve_repo_path(self.repo_root, config["data"].get("split_manifest"))
            if manifest_path is not None:
                manifest = load_split_manifest(manifest_path)
                self.artifacts.update(
                    dataset_identity={
                        "manifest_path": str(config["data"]["split_manifest"]),
                        "manifest_sha256": sha256_file(manifest_path),
                        "dataset_id": manifest["dataset_id"],
                        "dataset_sha256": manifest["dataset_sha256"],
                        "split_counts": manifest["split_counts"],
                        "split_fingerprints": manifest["split_fingerprints"],
                        "processed_split_fingerprints": metadata["processed_split_fingerprints"],
                        "processed_chart_type": metadata.get("chart_type"),
                        "processed_align_to_last_heading": bool(metadata.get("align_to_last_heading", False)),
                        "protocol": manifest["protocol"],
                    }
                )
            chart_type = metadata.get("chart_type", config["data"].get("chart_type", "position"))
            thresholds = json.loads(summary.thresholds_path.read_text()) if summary.thresholds_path.exists() else _default_thresholds()
            train_loader, val_loader, test_loader = self._build_loaders(summary.processed_root)
            model = build_model(config).to(self.device)
            parameter_count = count_trainable_parameters(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"]) if parameter_count > 0 else None
            scheduler = ReduceLROnPlateau(optimizer, factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"]) if optimizer is not None else None
            position_norm = self._position_normalization(train_loader)
            chart_norm = self._chart_normalization(summary) if self.uses_chart_model else None
            self.artifacts.update(model_metadata=model_metadata(model, config), model_parameter_count=parameter_count, dataset_split_sizes=summary.split_sizes)

            best_val_ade = float("inf")
            best_val_fde = float("inf")
            best_epoch = 0
            epochs_without_improvement = 0
            last_epoch = 0
            trainable = optimizer is not None
            epochs = int(config["training"]["epochs"]) if trainable else 0
            if trainable:
                for epoch in range(1, epochs + 1):
                    started = time.time()
                    train_metrics = self._run_epoch(model, train_loader, optimizer, thresholds, position_norm, chart_norm, chart_type, train=True)
                    val_metrics = self._run_epoch(model, val_loader, None, thresholds, position_norm, chart_norm, chart_type, train=False)
                    if scheduler is not None:
                        scheduler.step(val_metrics["ade"])
                    record = {"epoch": epoch, "seconds": time.time() - started, "train": train_metrics, "val": val_metrics}
                    last_epoch = epoch
                    self.artifacts.record_epoch(record)
                    self.logger.info("epoch=%s train_total=%.4f val_ade=%.4f", epoch, train_metrics["total"], val_metrics["ade"])
                    if val_metrics["ade"] < best_val_ade:
                        best_val_ade = val_metrics["ade"]
                        best_val_fde = val_metrics["fde"]
                        best_epoch = epoch
                        epochs_without_improvement = 0
                        self.artifacts.save_checkpoint("best.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, selection={"best_validation_ade": best_val_ade, "best_validation_fde": best_val_fde})
                        self.artifacts.update(best_epoch=best_epoch, best_validation_ade=best_val_ade, best_validation_fde=best_val_fde)
                    else:
                        epochs_without_improvement += 1
                    self.artifacts.save_periodic_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, interval=config["training"]["checkpoint_every_epochs"], selection={"best_validation_ade": best_val_ade})
                    if epochs_without_improvement > 0 and epochs_without_improvement >= config["training"]["early_stopping_patience"]:
                        break
            else:
                val_metrics, _ = self.evaluate_split(model, val_loader, thresholds, position_norm, chart_norm, chart_type)
                best_val_ade = val_metrics["ade"]
                best_val_fde = val_metrics["fde"]
                self.artifacts.record_epoch({"epoch": 0, "seconds": 0.0, "train": {"total": 0.0, "ade": 0.0, "fde": 0.0, "position_mse": 0.0}, "val": val_metrics})
                self.artifacts.save_checkpoint("best.pt", model=model, optimizer=None, scheduler=None, epoch=0, selection={"best_validation_ade": best_val_ade, "best_validation_fde": best_val_fde})
                self.artifacts.update(best_epoch=0, best_validation_ade=best_val_ade, best_validation_fde=best_val_fde)

            final_epoch = best_epoch if not trainable else last_epoch
            self.artifacts.save_checkpoint("final.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=final_epoch, selection={"best_validation_ade": best_val_ade, "best_validation_fde": best_val_fde})
            best_checkpoint = torch.load(self.checkpoint_dir / "best.pt", map_location=self.device, weights_only=False)
            model.load_state_dict(best_checkpoint["model_state"])
            val_summary, val_outputs = self.evaluate_split(model, val_loader, thresholds, position_norm, chart_norm, chart_type)
            evaluate_test = bool(config["evaluation"].get("evaluate_test", True))
            if evaluate_test:
                assert test_loader is not None
                final_split = "test"
                final_summary, final_outputs = self.evaluate_split(model, test_loader, thresholds, position_norm, chart_norm, chart_type)
            else:
                final_split = "validation"
                final_summary, final_outputs = val_summary, val_outputs
            metrics_json_path, _ = self._save_metrics(final_summary, final_split)
            plot_loss_curves(self.metrics_jsonl, self.figure_dir / "loss_curves.png")
            selected_examples = plot_trajectory_examples(final_outputs["history"], final_outputs["future"], final_outputs["predicted"], self.figure_dir / "trajectory_examples.png", max_examples=int(config["evaluation"]["save_prediction_examples"]), sample_ids=final_outputs["sample_ids"], seed=config["seed"])
            prediction_summary_path = self.evaluation_dir / f"{final_split}_predictions_summary.json"
            write_json(
                prediction_summary_path,
                {
                    "split": final_split,
                    "num_samples": len(final_outputs["sample_ids"]),
                    "example_sample_count": len(selected_examples),
                    "example_sample_ids": [final_outputs["sample_ids"][index] for index in selected_examples],
                },
            )
            self.artifacts.complete(
                model=config["model"]["name"],
                seed=config["seed"],
                model_metadata=model_metadata(model, config),
                model_parameter_count=parameter_count,
                dataset_split_sizes=summary.split_sizes,
                train_derived_normalization_statistics_path=str(summary.normalization_path),
                train_derived_feasibility_thresholds=thresholds,
                best_epoch=best_epoch,
                best_validation_ade=val_summary["ade"],
                best_validation_fde=val_summary["fde"],
                final_validation_ade=val_summary["ade"],
                final_validation_fde=val_summary["fde"],
                test_evaluated=evaluate_test,
                test_ade=final_summary["ade"] if evaluate_test else None,
                test_fde=final_summary["fde"] if evaluate_test else None,
                runtime_seconds=time.time() - started_run,
                checkpoint_paths={
                    "best": str(self.checkpoint_dir / "best.pt"),
                    "final": str(self.checkpoint_dir / "final.pt"),
                    "periodic": sorted(str(path) for path in self.checkpoint_dir.glob("epoch_*.pt")),
                },
                evaluation_split=final_split,
                evaluation_metrics_path=str(metrics_json_path),
                prediction_summary_path=str(prediction_summary_path),
            )
            return self.result_dir
        except KeyboardInterrupt:
            self.artifacts.update(status="interrupted")
            raise
        except Exception as exc:
            self.artifacts.fail(exc)
            raise
