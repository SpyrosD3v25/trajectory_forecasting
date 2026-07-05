import torch

from kineroute_nvp.losses.trajectory_physics import trajectory_loss


def test_losses_are_finite() -> None:
    history = torch.randn(4, 30, 2)
    future = torch.randn(4, 30, 2)
    predicted = future + 0.1
    losses = trajectory_loss(
        predicted_future=predicted,
        target_future=future,
        history_positions=history,
        dt_seconds=20.0,
        lambda_velocity=0.1,
        lambda_acceleration=0.05,
        lambda_turn=0.05,
        lambda_feasibility=0.01,
        max_acceleration=1.0,
        max_turn_rate=0.5,
    )
    for key in ["total", "ade", "fde", "position_mse", "velocity", "acceleration", "turn", "feasibility"]:
        assert torch.isfinite(losses[key]).item()
