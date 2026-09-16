from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kineroute_nvp.data.dataset import ProcessedTrajectoryDataset
from kineroute_nvp.data.split_manifest import validate_experiment_protocol, validate_processed_dataset
from kineroute_nvp.data.split_manifest import sha256_file
from kineroute_nvp.data.trajectory_preprocessing import decode_with_history_torch, encode_with_history_torch
from kineroute_nvp.evaluation.metrics import summarize_predictions
from kineroute_nvp.geometry.kinematics import decode_chart_torch
from kineroute_nvp.models.baselines import DeadReckoningBaseline
from kineroute_nvp.models.factory import build_model
from kineroute_nvp.training.trainer import CHART_MODELS
from kineroute_nvp.utils.io import write_json
from kineroute_nvp.utils.model import count_trainable_parameters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a checkpoint and evaluate one split once.")
    parser.add_argument("checkpoint", help="Checkpoint path, or a result directory containing checkpoints/best.pt.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _resolve_checkpoint(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_dir():
        return path / "checkpoints" / "best.pt"
    return path


def _load_chart_norm(processed_root: Path, device: torch.device):
    normalization = json.loads((processed_root / "normalization.json").read_text())
    if "history_chart_mean" in normalization:
        return (
            torch.tensor(normalization["history_chart_mean"], dtype=torch.float32, device=device),
            torch.tensor(normalization["history_chart_std"], dtype=torch.float32, device=device),
            torch.tensor(normalization["future_chart_mean"], dtype=torch.float32, device=device),
            torch.tensor(normalization["future_chart_std"], dtype=torch.float32, device=device),
        )
    mean = torch.tensor(normalization["chart_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(normalization["chart_std"], dtype=torch.float32, device=device)
    return mean, std, mean, std


def _position_norm(dataset: ProcessedTrajectoryDataset, device: torch.device, enabled: bool, mode: str):
    if not enabled:
        return None
    history_metric = torch.as_tensor(dataset.history, dtype=torch.float32, device=device)
    future_metric = torch.as_tensor(dataset.future, dtype=torch.float32, device=device)
    history = encode_with_history_torch(history_metric, history_metric, mode)
    future = encode_with_history_torch(history_metric, future_metric, mode)
    return (
        history.mean(dim=0, keepdim=True).to(device),
        history.std(dim=0, keepdim=True, correction=0).clamp_min(1e-6).to(device),
        future.mean(dim=0, keepdim=True).to(device),
        future.std(dim=0, keepdim=True, correction=0).clamp_min(1e-6).to(device),
    )


def evaluate_checkpoint(checkpoint_path: Path, split: str, device: torch.device) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["resolved_config"]
    processed_root = Path(config["data"]["processed_root"])
    if not processed_root.is_absolute():
        processed_root = REPO_ROOT / processed_root
    metadata = json.loads((processed_root / "metadata.json").read_text())
    manifest_value = config["data"].get("split_manifest")
    manifest = None
    if manifest_value:
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        manifest = validate_processed_dataset(processed_root, manifest_path, config)
    validate_experiment_protocol(config, metadata, manifest)
    chart_type = metadata.get("chart_type", config["data"].get("chart_type", "position"))
    thresholds_path = processed_root / "feasibility_thresholds.json"
    thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {"max_acceleration": 1e9, "max_turn_rate": 1e9}
    train_dataset = ProcessedTrajectoryDataset(processed_root, config["data"]["train_split"], config["data"].get("max_train_samples"))
    dataset = ProcessedTrajectoryDataset(processed_root, split, config["data"].get(f"max_{split}_samples"))
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    uses_chart = config["model"]["name"] in CHART_MODELS
    chart_norm = _load_chart_norm(processed_root, device) if uses_chart else None
    mode = config["data"]["preprocessing_mode"]
    pos_norm = None if uses_chart or isinstance(model, DeadReckoningBaseline) else _position_norm(train_dataset, device, config["loss"].get("normalize", True), mode)
    histories = []
    futures = []
    predictions = []
    reverse_histories = []
    inversion_error = 0.0
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            if uses_chart:
                assert chart_norm is not None
                _, _, future_mean, future_std = chart_norm
                pred_chart = model(batch["history_chart"].to(device)) * future_std + future_mean
                predicted = decode_chart_torch(pred_chart, chart_type)
                reverse = history
                if hasattr(model, "inverse") and config["evaluation"].get("evaluate_reverse_mapping", True):
                    history_mean, history_std, _, _ = chart_norm
                    future_chart = batch["future_chart"].to(device)
                    reverse = decode_chart_torch(model.inverse(future_chart) * history_std + history_mean, chart_type)
                    history_chart = batch["history_chart"].to(device)
                    inversion_error = max(inversion_error, float(torch.max(torch.abs(model.inverse(model(history_chart)) - history_chart)).item()))
            elif pos_norm is not None:
                history_mean, history_std, future_mean, future_std = pos_norm
                encoded_history = encode_with_history_torch(history, history, mode)
                predicted_encoded = model((encoded_history - history_mean) / history_std) * future_std + future_mean
                predicted = decode_with_history_torch(history, predicted_encoded, mode)
                reverse = history
            else:
                if isinstance(model, DeadReckoningBaseline):
                    predicted = model(history)
                else:
                    predicted = decode_with_history_torch(history, model(encode_with_history_torch(history, history, mode)), mode)
                reverse = history
            histories.append(history.cpu())
            futures.append(future.cpu())
            predictions.append(predicted.cpu())
            reverse_histories.append(reverse.cpu())
    history_tensor = torch.cat(histories)
    future_tensor = torch.cat(futures)
    predicted_tensor = torch.cat(predictions)
    reverse_tensor = torch.cat(reverse_histories)
    metrics = summarize_predictions(
        history_tensor,
        predicted_tensor,
        future_tensor,
        reverse_tensor,
        config["data"]["dt_seconds"],
        thresholds["max_acceleration"],
        thresholds["max_turn_rate"],
    )
    metrics["invertibility_max_abs"] = inversion_error
    metrics["model_parameter_count"] = float(count_trainable_parameters(model))
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    metrics = evaluate_checkpoint(checkpoint_path, args.split, torch.device(args.device))
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        write_json(output_dir / "metrics.json", metrics)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        write_json(
            output_dir / "provenance.json",
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "split": args.split,
                "run_provenance": checkpoint.get("run_provenance", {}),
            },
        )
        with (output_dir / "metrics.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
