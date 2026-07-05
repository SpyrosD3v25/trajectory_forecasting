from __future__ import annotations

from typing import Dict

import torch

from kineroute_nvp.geometry.kinematics import build_sequence, sequence_kinematics
from kineroute_nvp.losses.trajectory_physics import ade, fde


def summarize_predictions(
    history: torch.Tensor,
    predicted_future: torch.Tensor,
    target_future: torch.Tensor,
    reverse_history: torch.Tensor,
    dt_seconds: float,
    max_acceleration: float,
    max_turn_rate: float,
) -> Dict[str, float]:
    last_history = history[:, -1, :]
    predicted_sequence = build_sequence(last_history, predicted_future)
    target_sequence = build_sequence(last_history, target_future)
    predicted_kin = sequence_kinematics(predicted_sequence, dt_seconds)
    target_kin = sequence_kinematics(target_sequence, dt_seconds)

    acc_norm = torch.linalg.norm(predicted_kin["acceleration"], dim=-1)
    turn_abs = torch.abs(predicted_kin["turn_rate"])
    return {
        "ade": float(ade(predicted_future, target_future).item()),
        "fde": float(fde(predicted_future, target_future).item()),
        "reverse_ade": float(ade(reverse_history, history).item()),
        "reverse_fde": float(fde(reverse_history, history).item()),
        "velocity_error": float(torch.linalg.norm(predicted_kin["velocity"] - target_kin["velocity"], dim=-1).mean().item()),
        "acceleration_error": float(torch.linalg.norm(predicted_kin["acceleration"] - target_kin["acceleration"], dim=-1).mean().item()),
        "turn_rate_error": float(torch.abs(predicted_kin["turn_rate"] - target_kin["turn_rate"]).mean().item()),
        "infeasible_acceleration_pct": float((acc_norm > max_acceleration).float().mean().item() * 100.0),
        "infeasible_turn_rate_pct": float((turn_abs > max_turn_rate).float().mean().item() * 100.0),
    }
