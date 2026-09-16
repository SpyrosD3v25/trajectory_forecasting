from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kineroute_nvp.data.dataset import ProcessedTrajectoryDataset
from kineroute_nvp.data.split_manifest import validate_experiment_protocol, validate_processed_dataset
from kineroute_nvp.data.trajectory_preprocessing import decode_with_history_torch, encode_with_history_torch
from kineroute_nvp.evaluation.plots import plot_trajectory_examples
from kineroute_nvp.geometry.kinematics import decode_chart_torch
from kineroute_nvp.models.baselines import DeadReckoningBaseline
from kineroute_nvp.models.factory import build_model
from kineroute_nvp.training.trainer import CHART_MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize history, target future, and predicted future from a checkpoint.")
    parser.add_argument("--pt", required=True, help="Path to a checkpoint `.pt` file relative to the project root.")
    parser.add_argument("--n", type=int, required=True, help="Number of samples to visualize.")
    parser.add_argument("--random", action="store_true", help="Sample random examples instead of taking the first N.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for reproducible random example selection.")
    parser.add_argument("--grid", action="store_true", help="Save one grid image instead of one file per sample.")
    parser.add_argument(
        "--output-dir",
        default="plots/trajectory_visualizations",
        help="Directory where rendered figures will be saved, relative to the project root.",
    )
    return parser


def _device_for_inference() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {"model_state", "resolved_config"}
    missing = sorted(required_keys - checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint {path} is missing required keys: {', '.join(missing)}")
    return checkpoint


def _processed_root_from_config(config: dict) -> Path:
    data_cfg = config.get("data", {})
    processed_root = data_cfg.get("processed_root")
    if not processed_root:
        raise ValueError("Checkpoint config is missing data.processed_root.")
    return REPO_ROOT / processed_root


def _split_name(config: dict, key: str, default: str) -> str:
    return config.get("data", {}).get(key, default)


def _validate_data(config: dict, processed_root: Path) -> dict:
    metadata = json.loads((processed_root / "metadata.json").read_text())
    manifest = None
    manifest_value = config.get("data", {}).get("split_manifest")
    if manifest_value:
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        manifest = validate_processed_dataset(processed_root, manifest_path, config)
    validate_experiment_protocol(config, metadata, manifest)
    return metadata


def _evaluate_baseline(checkpoint: dict, config: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    processed_root = _processed_root_from_config(config)
    _validate_data(config, processed_root)
    train_split = _split_name(config, "train_split", "train")
    test_split = _split_name(config, "test_split", "test")
    train_dataset = ProcessedTrajectoryDataset(processed_root, train_split, config.get("data", {}).get("max_train_samples"))
    test_dataset = ProcessedTrajectoryDataset(processed_root, test_split, config.get("data", {}).get("max_test_samples"))
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loss_cfg = config.get("loss", {})
    normalize = bool(loss_cfg.get("normalize", True))
    mode = config["data"]["preprocessing_mode"]
    if normalize and not isinstance(model, DeadReckoningBaseline):
        train_history_metric = torch.from_numpy(train_dataset.history)
        train_future_metric = torch.from_numpy(train_dataset.future)
        train_history = encode_with_history_torch(train_history_metric, train_history_metric, mode)
        train_future = encode_with_history_torch(train_history_metric, train_future_metric, mode)
        history_mean = train_history.mean(dim=0, keepdim=True).to(device)
        history_std = train_history.std(dim=0, keepdim=True, correction=0).clamp_min(1e-6).to(device)
        future_mean = train_future.mean(dim=0, keepdim=True).to(device)
        future_std = train_future.std(dim=0, keepdim=True, correction=0).clamp_min(1e-6).to(device)

    histories = []
    futures = []
    preds = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            if isinstance(model, DeadReckoningBaseline):
                pred = model(history)
            elif normalize:
                encoded_history = encode_with_history_torch(history, history, mode)
                norm_history = (encoded_history - history_mean) / history_std
                pred_norm = model(norm_history)
                pred = decode_with_history_torch(history, pred_norm * future_std + future_mean, mode)
            else:
                encoded_history = encode_with_history_torch(history, history, mode)
                pred = decode_with_history_torch(history, model(encoded_history), mode)
            histories.append(history.cpu().numpy())
            futures.append(future.cpu().numpy())
            preds.append(pred.cpu().numpy())
            sample_ids.extend(batch["sample_id"])
    return np.concatenate(histories), np.concatenate(futures), np.concatenate(preds), sample_ids


def _evaluate_kineroute(checkpoint: dict, config: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    processed_root = _processed_root_from_config(config)
    metadata = _validate_data(config, processed_root)
    test_split = _split_name(config, "test_split", "test")
    test_dataset = ProcessedTrajectoryDataset(processed_root, test_split, config.get("data", {}).get("max_test_samples"))
    normalization = json.loads((processed_root / "normalization.json").read_text())
    chart_type = metadata.get("chart_type", "polar")
    if "history_chart_mean" in normalization:
        future_chart_mean = torch.tensor(normalization["future_chart_mean"], dtype=torch.float32, device=device)
        future_chart_std = torch.tensor(normalization["future_chart_std"], dtype=torch.float32, device=device)
    else:
        shared_mean = torch.tensor(normalization["chart_mean"], dtype=torch.float32, device=device)
        shared_std = torch.tensor(normalization["chart_std"], dtype=torch.float32, device=device)
        future_chart_mean = shared_mean
        future_chart_std = shared_std

    loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    histories = []
    futures = []
    preds = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            predicted_chart_norm = model(batch["history_chart"].to(device))
            predicted_chart = predicted_chart_norm * future_chart_std + future_chart_mean
            predicted_future = decode_chart_torch(predicted_chart, chart_type)
            histories.append(history.cpu().numpy())
            futures.append(future.cpu().numpy())
            preds.append(predicted_future.cpu().numpy())
            sample_ids.extend(batch["sample_id"])
    return np.concatenate(histories), np.concatenate(futures), np.concatenate(preds), sample_ids


def _select_indices(total: int, count: int, randomize: bool, seed: int = 42) -> np.ndarray:
    if total == 0:
        raise ValueError("No samples available to visualize.")
    clamped = min(count, total)
    if randomize:
        return np.random.default_rng(seed).choice(total, size=clamped, replace=False)
    return np.arange(clamped)


def _axis_limits(history: np.ndarray, future: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float, float]:
    all_points = np.concatenate([history, future, predicted], axis=0)
    min_x = float(all_points[:, 0].min())
    max_x = float(all_points[:, 0].max())
    min_y = float(all_points[:, 1].min())
    max_y = float(all_points[:, 1].max())
    pad_x = max((max_x - min_x) * 0.05, 1e-3)
    pad_y = max((max_y - min_y) * 0.05, 1e-3)
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def _plot_sample(history: np.ndarray, future: np.ndarray, predicted: np.ndarray, sample_id: str, sample_idx: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    min_x, max_x, min_y, max_y = _axis_limits(history, future, predicted)
    panels = (
        ("Past 30", history, "tab:blue"),
        ("True Next 30", future, "tab:green"),
        ("Predicted Next 30", predicted, "tab:red"),
    )
    for ax, (title, series, color) in zip(axes, panels):
        ax.plot(series[:, 0], series[:, 1], color=color, marker="o", markersize=2)
        ax.scatter(series[0, 0], series[0, 1], color="black", s=18)
        ax.set_title(title)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Sample {sample_idx}: {sample_id}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def _should_show_figures() -> bool:
    return "agg" not in matplotlib.get_backend().lower()


def _checkpoint_output_dir(base_output_dir: Path, checkpoint_path: Path) -> Path:
    if base_output_dir.name == "figures":
        base_output_dir.mkdir(parents=True, exist_ok=True)
        return base_output_dir
    run_name = checkpoint_path.parent.parent.name if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.stem
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n <= 0:
        raise SystemExit("--n must be a positive integer.")

    checkpoint_path = REPO_ROOT / args.pt
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    output_root = REPO_ROOT / args.output_dir

    checkpoint = _load_checkpoint(checkpoint_path)
    config = checkpoint["resolved_config"]
    model_name = config.get("model", {}).get("name")
    device = _device_for_inference()

    if model_name in CHART_MODELS:
        history, future, predicted, sample_ids = _evaluate_kineroute(checkpoint, config, device)
    else:
        history, future, predicted, sample_ids = _evaluate_baseline(checkpoint, config, device)

    indices = _select_indices(len(sample_ids), args.n, args.random, args.seed)
    output_dir = _checkpoint_output_dir(output_root, checkpoint_path)
    if args.grid:
        output_path = output_dir / "trajectory_examples.png"
        selected_ids = [sample_ids[index] for index in indices]
        plot_trajectory_examples(
            history[indices],
            future[indices],
            predicted[indices],
            output_path,
            max_examples=len(indices),
            sample_ids=selected_ids,
            seed=args.seed,
        )
        manifest_path = output_dir / "trajectory_examples_samples.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "split": _split_name(config, "test_split", "test"),
                    "seed": args.seed,
                    "sample_ids": selected_ids,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Saved grid: {output_path.relative_to(REPO_ROOT)}")
        print(f"Saved sample manifest: {manifest_path.relative_to(REPO_ROOT)}")
        return 0
    for figure_idx, sample_idx in enumerate(indices, start=1):
        fig = _plot_sample(history[sample_idx], future[sample_idx], predicted[sample_idx], sample_ids[sample_idx], figure_idx)
        output_path = output_dir / f"sample_{figure_idx:03d}__{sample_ids[sample_idx]}.png"
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot: {output_path.relative_to(REPO_ROOT)}")
    if _should_show_figures():
        plt.show()
    else:
        plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
