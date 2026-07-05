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


