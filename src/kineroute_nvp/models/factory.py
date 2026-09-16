from __future__ import annotations

from typing import Any

import torch

from kineroute_nvp.models.baselines import (
    DeadReckoningBaseline,
    GRUBaseline,
    LSTMBaseline,
    MLPForecaster,
    ResidualMLPForecaster,
    Seq2SeqBaseline,
    TCNForecaster,
    TransformerNARForecaster,
)
from kineroute_nvp.models.kineroute_nvp import KineRouteNVP, NonInvertibleRoutedControl, StandardRealNVP


def _configured(model_cfg: dict[str, Any], key: str, default: Any) -> Any:
    value = model_cfg.get(key)
    return default if value is None else value


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    name = model_cfg["name"]
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", model_cfg.get("hidden_layers", 2)))
    hidden_layers = int(model_cfg.get("hidden_layers", num_layers))
    if name == "dead_reckoning":
        return DeadReckoningBaseline(
            velocity_points=int(model_cfg.get("velocity_points", 2)),
            future_steps=int(config["data"].get("future_steps", 30)),
        )
    if name == "gru":
        return GRUBaseline(hidden_dim=hidden_dim, num_layers=num_layers, bidirectional=False)
    if name == "bigru":
        return GRUBaseline(hidden_dim=hidden_dim, num_layers=num_layers, bidirectional=True)
    if name == "lstm":
        return LSTMBaseline(hidden_dim=hidden_dim, num_layers=num_layers, bidirectional=False)
    if name == "bilstm":
        return LSTMBaseline(hidden_dim=hidden_dim, num_layers=num_layers, bidirectional=True)
    if name == "seq2seq":
        return Seq2SeqBaseline(hidden_dim=hidden_dim, num_layers=num_layers)
    if name == "mlp":
        return MLPForecaster(hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    if name == "resnet":
        return ResidualMLPForecaster(hidden_dim=hidden_dim, residual_blocks=int(_configured(model_cfg, "residual_blocks", hidden_layers)))
    if name == "tcn":
        return TCNForecaster(
            hidden_dim=hidden_dim,
            levels=int(_configured(model_cfg, "levels", hidden_layers)),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
        )
    if name == "transformer_nar":
        return TransformerNARForecaster(
            d_model=int(_configured(model_cfg, "d_model", hidden_dim)),
            num_layers=num_layers,
            num_heads=int(model_cfg.get("num_heads", 4)),
            dim_feedforward=int(_configured(model_cfg, "dim_feedforward", hidden_dim * 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name in {"kineroute_nvp", "routed_realnvp"}:
        return KineRouteNVP(
            num_routed_blocks=int(_configured(model_cfg, "num_routed_blocks", _configured(model_cfg, "num_blocks", 6))),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            scale_bound=float(model_cfg.get("scale_bound", 1.5)),
        )
    if name == "realnvp":
        return StandardRealNVP(
            num_blocks=int(_configured(model_cfg, "num_blocks", _configured(model_cfg, "num_routed_blocks", 6))),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            scale_bound=float(model_cfg.get("scale_bound", 1.5)),
        )
    if name == "non_invertible_routed":
        return NonInvertibleRoutedControl(
            num_blocks=int(_configured(model_cfg, "num_blocks", _configured(model_cfg, "num_routed_blocks", 6))),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
    raise ValueError(f"Unsupported model name: {name}")


def model_metadata(model: torch.nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config["model"]["name"],
        "class_name": model.__class__.__name__,
        "hyperparameters": dict(config["model"]),
    }
