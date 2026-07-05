from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kineroute_nvp.data.dataset import ProcessedTrajectoryDataset
from kineroute_nvp.data.preprocess import prepare_envship_dataset
from kineroute_nvp.evaluation.metrics import summarize_predictions
from kineroute_nvp.evaluation.plots import plot_trajectory_examples
from kineroute_nvp.geometry.kinematics import build_sequence
from kineroute_nvp.losses.trajectory_physics import ade, fde
from kineroute_nvp.models.baselines import GRUBaseline, LSTMBaseline, Seq2SeqBaseline
from kineroute_nvp.utils.baseline_config import args_to_config, parse_args
from kineroute_nvp.utils.io import append_jsonl, atomic_write_json, environment_snapshot, get_git_commit, write_json, write_yaml
from kineroute_nvp.utils.seed import set_seed


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model_name == "gru":
        return GRUBaseline(hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=False)
    if args.model_name == "bigru":
        return GRUBaseline(hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=True)
    if args.model_name == "lstm":
        return LSTMBaseline(hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=False)
    if args.model_name == "bilstm":
        return LSTMBaseline(hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidirectional=True)
    return Seq2SeqBaseline(hidden_dim=args.hidden_dim, num_layers=args.num_layers)


def evaluate(model, loader, device) -> tuple[dict, dict]:
    histories, futures, preds = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            pred = model(history)
            histories.append(history.cpu())
            futures.append(future.cpu())
            preds.append(pred.cpu())
    history = torch.cat(histories)
    future = torch.cat(futures)
    pred = torch.cat(preds)
    last_history = history[:, -1, :]
    pred_seq = build_sequence(last_history, pred)
    true_seq = build_sequence(last_history, future)
    metrics = summarize_predictions(
        history=history,
        predicted_future=pred,
        target_future=future,
        reverse_history=history,
        dt_seconds=20.0,
        max_acceleration=1e9,
        max_turn_rate=1e9,
    )
    metrics["ade"] = float(ade(pred, future).item())
    metrics["fde"] = float(fde(pred, future).item())
    metrics["reverse_ade"] = 0.0
    metrics["reverse_fde"] = 0.0
    return metrics, {"history": history.numpy(), "future": future.numpy(), "predicted": pred.numpy()}


def evaluate_normalized(model, loader, device, history_mean, history_std, future_mean, future_std) -> tuple[dict, dict]:
    histories, futures, preds = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            norm_history = (history - history_mean) / history_std
            pred_norm = model(norm_history)
            pred = pred_norm * future_std + future_mean
            histories.append(history.cpu())
            futures.append(future.cpu())
            preds.append(pred.cpu())
    history = torch.cat(histories)
    future = torch.cat(futures)
    pred = torch.cat(preds)
    metrics = summarize_predictions(
        history=history,
        predicted_future=pred,
        target_future=future,
        reverse_history=history,
        dt_seconds=20.0,
        max_acceleration=1e9,
        max_turn_rate=1e9,
    )
    metrics["ade"] = float(ade(pred, future).item())
    metrics["fde"] = float(fde(pred, future).item())
    metrics["reverse_ade"] = 0.0
    metrics["reverse_fde"] = 0.0
    return metrics, {"history": history.numpy(), "future": future.numpy(), "predicted": pred.numpy()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = args_to_config(args)
    set_seed(args.seed)

    processed_root = REPO_ROOT / args.processed_root
    if not (processed_root / "metadata.json").exists():
        prepare_envship_dataset(
            raw_root=REPO_ROOT / args.raw_root if args.raw_root else None,
            raw_roots=[REPO_ROOT / root for root in (args.raw_roots or [])],
            processed_root=processed_root,
            dt_seconds=20.0,
            feasibility_quantile=0.995,
            align_to_last_heading=False,
            chart_type="position",
            splits=("train", "val", "test"),
            include_ship_classes=args.include_ship_classes,
            quality_tiers=args.quality_tiers,
        )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataset = ProcessedTrajectoryDataset(processed_root, "train", args.max_train_samples)
    val_dataset = ProcessedTrajectoryDataset(processed_root, "val", args.max_val_samples)
    test_dataset = ProcessedTrajectoryDataset(processed_root, "test", args.max_test_samples)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=8)

    history_mean = history_std = future_mean = future_std = None
    if args.normalize:
        train_history = torch.from_numpy(train_dataset.history)
        train_future = torch.from_numpy(train_dataset.future)
        history_mean = train_history.mean(dim=(0, 1), keepdim=True).to(device)
        history_std = train_history.std(dim=(0, 1), keepdim=True).clamp_min(1e-6).to(device)
        future_mean = train_future.mean(dim=(0, 1), keepdim=True).to(device)
        future_std = train_future.std(dim=(0, 1), keepdim=True).clamp_min(1e-6).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = REPO_ROOT / args.results_root / f"{timestamp}__{args.experiment_name}__{args.run_name}"
    (result_dir / "logs").mkdir(parents=True, exist_ok=False)
    (result_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (result_dir / "figures").mkdir(parents=True, exist_ok=True)
    write_yaml(result_dir / "resolved_config.yaml", config)
    write_json(result_dir / "environment.json", environment_snapshot())
    (result_dir / "command.txt").write_text("python scripts/train_baseline.py " + " ".join(shlex.quote(x) for x in (argv or sys.argv[1:])) + "\n")

    results_path = result_dir / "_results.json"
    atomic_write_json(results_path, {"status": "running", "model": args.model_name, "run_name": args.run_name, "seed": args.seed})

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    last_completed_epoch = 0
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            count = 0
            for batch in train_loader:
                history = batch["history"].to(device)
                future = batch["future"].to(device)
                if args.normalize:
                    norm_history = (history - history_mean) / history_std
                    norm_future = (future - future_mean) / future_std
                    if args.model_name == "seq2seq":
                        pred_norm = model(norm_history, norm_future, teacher_forcing_ratio=args.teacher_forcing_ratio)
                    else:
                        pred_norm = model(norm_history)
                    if args.loss_name == "mse":
                        loss = F.mse_loss(pred_norm, norm_future)
                    else:
                        pred = pred_norm * future_std + future_mean
                        loss = ade(pred, future)
                else:
                    if args.model_name == "seq2seq":
                        pred = model(history, future, teacher_forcing_ratio=args.teacher_forcing_ratio)
                    else:
                        pred = model(history)
                    loss = F.mse_loss(pred, future) if args.loss_name == "mse" else ade(pred, future)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * history.shape[0]
                count += history.shape[0]
            train_ade = running / count
            if args.normalize:
                val_metrics, _ = evaluate_normalized(model, val_loader, device, history_mean, history_std, future_mean, future_std)
            else:
                val_metrics, _ = evaluate(model, val_loader, device)
            scheduler.step(val_metrics["ade"])
            append_jsonl(result_dir / "logs" / "metrics.jsonl", {"epoch": epoch, "train_ade": train_ade, "val": val_metrics})
            print(f"epoch={epoch} train_ade={train_ade:.4f} val_ade={val_metrics['ade']:.4f}")
            last_completed_epoch = epoch
            if val_metrics["ade"] < best_val:
                best_val = val_metrics["ade"]
                best_epoch = epoch
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    except KeyboardInterrupt:
        atomic_write_json(
            results_path,
            {
                "status": "interrupted",
                "model": args.model_name,
                "run_name": args.run_name,
                "seed": args.seed,
                "last_completed_epoch": last_completed_epoch,
                "best_epoch": best_epoch if best_state is not None else None,
                "best_validation_ade": best_val if best_state is not None else None,
                "git_commit": get_git_commit(REPO_ROOT),
            },
        )
        return 130
    except Exception:
        atomic_write_json(
            results_path,
            {
                "status": "failed",
                "model": args.model_name,
                "run_name": args.run_name,
                "seed": args.seed,
                "last_completed_epoch": last_completed_epoch,
                "best_epoch": best_epoch if best_state is not None else None,
                "best_validation_ade": best_val if best_state is not None else None,
                "git_commit": get_git_commit(REPO_ROOT),
            },
        )
        raise

    assert best_state is not None
    model.load_state_dict(best_state)
    if args.normalize:
        val_metrics, _ = evaluate_normalized(model, val_loader, device, history_mean, history_std, future_mean, future_std)
        test_metrics, test_outputs = evaluate_normalized(model, test_loader, device, history_mean, history_std, future_mean, future_std)
    else:
        val_metrics, _ = evaluate(model, val_loader, device)
        test_metrics, test_outputs = evaluate(model, test_loader, device)
    plot_trajectory_examples(
        test_outputs["history"],
        test_outputs["future"],
        test_outputs["predicted"],
        result_dir / "figures" / "trajectory_examples.png",
        max_examples=24,
    )
    write_json(result_dir / "evaluation" / "metrics.json", test_metrics)
    atomic_write_json(
        results_path,
        {
            "status": "completed",
            "model": args.model_name,
            "run_name": args.run_name,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation_ade": val_metrics["ade"],
            "best_validation_fde": val_metrics["fde"],
            "test_ade": test_metrics["ade"],
            "test_fde": test_metrics["fde"],
            "git_commit": get_git_commit(REPO_ROOT),
        },
    )
    print(result_dir)
    return 0
