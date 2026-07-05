import numpy as np

from kineroute_nvp.geometry.kinematics import (
    chart_to_positions_numpy,
    decode_chart_numpy,
    encode_positions_numpy,
    positions_to_chart_numpy,
)


def test_chart_round_trip_handles_zero_steps() -> None:
    positions = np.zeros((2, 30, 2), dtype=np.float64)
    positions[0, :, 0] = np.linspace(0.0, 29.0, 30)
    positions[1, 5:, 0] = np.linspace(0.0, 24.0, 25)
    chart = positions_to_chart_numpy(positions)
    reconstructed = chart_to_positions_numpy(chart)
    np.testing.assert_allclose(reconstructed, positions, atol=1e-8)


def test_displacement_chart_round_trip() -> None:
    positions = np.random.randn(3, 30, 2).astype(np.float64)
    chart = encode_positions_numpy(positions, chart_type="displacement")
    reconstructed = decode_chart_numpy(chart, chart_type="displacement")
    np.testing.assert_allclose(reconstructed, positions, atol=1e-8)


def test_position_chart_round_trip() -> None:
    positions = np.random.randn(3, 30, 2).astype(np.float64)
    chart = encode_positions_numpy(positions, chart_type="position")
    reconstructed = decode_chart_numpy(chart, chart_type="position")
    np.testing.assert_allclose(reconstructed, positions, atol=1e-8)
