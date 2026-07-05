from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from kineroute_nvp.geometry.kinematics import build_sequence, sequence_kinematics


def ade(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(predicted - target, dim=-1).mean()


def fde(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(predicted[:, -1, :] - target[:, -1, :], dim=-1).mean()


def trajectory_loss(
    predicted_future: torch.Tensor,
    target_future: torch.Tensor,
    history_positions: torch.Tensor,
    dt_seconds: float,
    lambda_velocity: float,
    lambda_acceleration: float,
    lambda_turn: float,
    lambda_feasibility: float,
    max_acceleration: float,
    max_turn_rate: float,
) -> Dict[str, torch.Tensor]:
    last_history_point = history_positions[:, -1, :]
    predicted_sequence = build_sequence(last_history_point, predicted_future)
    target_sequence = build_sequence(last_history_point, target_future)

    predicted_kin = sequence_kinematics(predicted_sequence, dt_seconds=dt_seconds)
    target_kin = sequence_kinematics(target_sequence, dt_seconds=dt_seconds)

    loss_ade = ade(predicted_future, target_future)
    loss_velocity = torch.linalg.norm(predicted_kin["velocity"] - target_kin["velocity"], dim=-1).mean()
    loss_acceleration = torch.linalg.norm(predicted_kin["acceleration"] - target_kin["acceleration"], dim=-1).mean()
    loss_turn = F.l1_loss(predicted_kin["turn_rate"], target_kin["turn_rate"])
    acc_norm = torch.linalg.norm(predicted_kin["acceleration"], dim=-1)
    turn_abs = torch.abs(predicted_kin["turn_rate"])
    loss_feasibility = (
        torch.relu(acc_norm - max_acceleration).pow(2).mean()
        + torch.relu(turn_abs - max_turn_rate).pow(2).mean()
    )
    total = (
        loss_ade
        + lambda_velocity * loss_velocity
        + lambda_acceleration * loss_acceleration
        + lambda_turn * loss_turn
        + lambda_feasibility * loss_feasibility
    )
    return {
        "total": total,
        "ade": loss_ade,
        "fde": fde(predicted_future, target_future),
        "position_mse": torch.mean((predicted_future - target_future) ** 2),
        "velocity": loss_velocity,
        "acceleration": loss_acceleration,
        "turn": loss_turn,
        "feasibility": loss_feasibility,
        "predicted_kinematics": predicted_kin,
        "target_kinematics": target_kin,
    }
