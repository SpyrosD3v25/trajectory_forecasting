import torch

from kineroute_nvp.models.kineroute_nvp import KineRouteNVP


def test_model_forward_inverse_round_trip() -> None:
    model = KineRouteNVP(num_routed_blocks=3, hidden_dim=32, hidden_layers=2, scale_bound=1.5)
    x = torch.randn(8, 60)
    reconstructed = model.inverse(model(x))
    assert torch.max(torch.abs(reconstructed - x)).item() < 1e-5
