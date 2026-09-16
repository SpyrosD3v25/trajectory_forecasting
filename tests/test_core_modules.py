import numpy as np
import torch

from kineroute_nvp.data.trajectory_preprocessing import (
    decode_with_history_torch,
    encode_history_future,
    encode_trajectory,
    encode_with_history_torch,
)
from kineroute_nvp.models.baselines import (
    DeadReckoningBaseline,
    GRUBaseline,
    LSTMBaseline,
    MLPForecaster,
    ResidualMLPForecaster,
    TCNForecaster,
    TransformerNARForecaster,
)
from kineroute_nvp.models.kineroute_nvp import NonInvertibleRoutedControl, StandardRealNVP
from kineroute_nvp.utils.model import count_trainable_parameters


def _trajectory() -> np.ndarray:
    steps = np.arange(30, dtype=np.float64)
    return np.stack([2.0 * steps + 5.0, 0.5 * steps**2 - 7.0], axis=-1)


def test_heading_aligned_displacement_round_trip() -> None:
    trajectory = _trajectory()
    encoded, transform = encode_trajectory(trajectory, "heading_aligned_displacement")
    decoded = transform.decode(encoded)

    assert encoded.shape == (30, 2)
    # The public representation is float32, so allow its expected rounding at
    # the scale of this synthetic trajectory.
    assert np.max(np.abs(decoded - trajectory)) < 3e-5


def test_history_future_transform_uses_history_heading() -> None:
    history = _trajectory()
    future = history + np.asarray([100.0, -20.0])
    encoded_history, encoded_future, transform = encode_history_future(history, future, "heading_aligned")

    assert encoded_history.shape == (30, 2)
    assert encoded_future.shape == (30, 2)
    np.testing.assert_allclose(transform.decode(encoded_future), future, atol=1e-5)


def test_heading_alignment_anchors_final_observation_and_is_translation_invariant() -> None:
    history = np.stack(
        [np.full(30, 12.0), np.linspace(-40.0, 18.0, 30)],
        axis=-1,
    )
    translated = history + np.asarray([900.0, -300.0])

    encoded, transform = encode_trajectory(history, "heading_aligned")
    translated_encoded, translated_transform = encode_trajectory(translated, "heading_aligned")

    np.testing.assert_allclose(transform.origin, history[-1])
    np.testing.assert_allclose(translated_transform.origin, translated[-1])
    np.testing.assert_allclose(encoded[-1], np.zeros(2), atol=1e-7)
    np.testing.assert_allclose(encoded, translated_encoded, atol=1e-5)
    assert encoded[-2, 0] < 0.0
    assert abs(float(encoded[-2, 1])) < 1e-6


def test_numpy_and_torch_history_derived_transforms_are_equivalent() -> None:
    history = _trajectory()
    future = _trajectory() + np.asarray([70.0, -12.0])
    history_batch = torch.from_numpy(history).float().unsqueeze(0)
    future_batch = torch.from_numpy(future).float().unsqueeze(0)

    for mode in ("raw", "heading_aligned", "heading_aligned_displacement"):
        numpy_history, numpy_future, _ = encode_history_future(history, future, mode)
        torch_history = encode_with_history_torch(history_batch, history_batch, mode)[0].numpy()
        torch_future = encode_with_history_torch(history_batch, future_batch, mode)[0].numpy()
        np.testing.assert_allclose(torch_history, numpy_history, atol=3e-5, rtol=1e-5)
        np.testing.assert_allclose(torch_future, numpy_future, atol=3e-5, rtol=1e-5)


def test_alignment_uses_last_valid_motion_when_final_observations_are_stationary() -> None:
    history = np.zeros((30, 2), dtype=np.float64)
    history[:21, 1] = np.arange(21, dtype=np.float64)
    history[21:, 1] = 20.0
    encoded, transform = encode_trajectory(history, "heading_aligned")

    assert transform.heading == np.pi / 2
    np.testing.assert_allclose(encoded[-1], np.zeros(2), atol=1e-7)
    np.testing.assert_allclose(encoded[19], np.asarray([-1.0, 0.0]), atol=1e-6)


def test_known_rotation_matches_expected_history_frame() -> None:
    steps = np.arange(30, dtype=np.float64)
    eastbound = np.stack([steps, np.zeros_like(steps)], axis=-1)
    angle = np.deg2rad(37.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_translated = eastbound @ rotation.T + np.asarray([50.0, -80.0])

    encoded, _ = encode_trajectory(rotated_translated, "heading_aligned")
    expected = np.stack([steps - steps[-1], np.zeros_like(steps)], axis=-1)

    np.testing.assert_allclose(encoded, expected, atol=1e-5)


def test_batched_torch_displacement_round_trip() -> None:
    history = torch.from_numpy(np.stack([_trajectory(), _trajectory() + 4.0])).float()
    future = history + torch.tensor([100.0, -20.0])

    encoded = encode_with_history_torch(history, future, "heading_aligned_displacement")
    decoded = decode_with_history_torch(history, encoded, "heading_aligned_displacement")

    torch.testing.assert_close(decoded, future, atol=1e-4, rtol=1e-5)


def test_dead_reckoning_extrapolates_constant_velocity_without_parameters() -> None:
    model = DeadReckoningBaseline(velocity_points=2)
    history = torch.zeros(2, 30, 2)
    history[:, :, 0] = torch.arange(30)
    history[:, :, 1] = 2.0 * torch.arange(30)

    prediction = model(history)

    assert count_trainable_parameters(model) == 0
    torch.testing.assert_close(prediction[0, 0], torch.tensor([30.0, 60.0]))
    torch.testing.assert_close(prediction[0, -1], torch.tensor([59.0, 118.0]))


def test_forecasters_emit_future_trajectory_shape_and_gradients() -> None:
    history = torch.randn(3, 30, 2)
    models = [
        MLPForecaster(hidden_dim=16, hidden_layers=1),
        ResidualMLPForecaster(hidden_dim=16, residual_blocks=1),
        TCNForecaster(hidden_dim=16, levels=2),
        TransformerNARForecaster(d_model=16, num_layers=1, num_heads=4, dim_feedforward=32, dropout=0.0),
    ]
    for model in models:
        prediction = model(history)
        assert prediction.shape == (3, 30, 2)
        loss = prediction.square().mean()
        loss.backward()
        assert count_trainable_parameters(model) > 0


def test_bidirectional_recurrent_heads_use_both_terminal_hidden_states() -> None:
    history = torch.randn(3, 30, 2)
    for model in (
        GRUBaseline(hidden_dim=8, num_layers=2, bidirectional=True),
        LSTMBaseline(hidden_dim=8, num_layers=2, bidirectional=True),
    ):
        model.eval()
        if isinstance(model, GRUBaseline):
            _, hidden = model.encoder(history)
        else:
            _, (hidden, _) = model.encoder(history)
        expected = model.head(torch.cat([hidden[-2], hidden[-1]], dim=-1)).view(3, 30, 2)
        torch.testing.assert_close(model(history), expected)


def test_standard_realnvp_round_trip() -> None:
    model = StandardRealNVP(num_blocks=3, hidden_dim=16, hidden_layers=1, scale_bound=1.5)
    x = torch.randn(4, 60)

    reconstructed = model.inverse(model(x))

    torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)


def test_non_invertible_routed_control_preserves_chart_shape() -> None:
    model = NonInvertibleRoutedControl(num_blocks=2, hidden_dim=16, hidden_layers=1)
    x = torch.randn(4, 60)

    output = model(x)

    assert output.shape == x.shape
    assert count_trainable_parameters(model) > 0


def test_non_invertible_routed_control_has_no_analytic_coupling_inverse() -> None:
    model = NonInvertibleRoutedControl(num_blocks=1, hidden_dim=16, hidden_layers=1)
    block = model.blocks[0]

    # Each A/S/T target network receives all 60 inputs, including its own target.
    # That breaks the triangular dependency required by the coupling inverse.
    assert block.a_net[0].in_features == 60
    assert block.s_net[0].in_features == 60
    assert block.t_net[0].in_features == 60
    assert not hasattr(block, "inverse")
    assert not hasattr(model, "inverse")
