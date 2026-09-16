import numpy as np

from kineroute_nvp.evaluation.plots import plot_trajectory_examples


def test_trajectory_examples_are_seeded_random_and_not_first_n(tmp_path):
    history = np.arange(40 * 30 * 2, dtype=np.float32).reshape(40, 30, 2)
    future = history + 1
    predicted = history + 2
    sample_ids = [f"sample-{index}" for index in range(len(history))]
    output_path = tmp_path / "examples.png"

    first_selection = plot_trajectory_examples(
        history,
        future,
        predicted,
        output_path,
        max_examples=24,
        sample_ids=sample_ids,
        seed=123,
    )
    second_selection = plot_trajectory_examples(
        history,
        future,
        predicted,
        output_path,
        max_examples=24,
        sample_ids=sample_ids,
        seed=123,
    )

    assert len(first_selection) == 24
    assert len(set(first_selection)) == 24
    assert first_selection == second_selection
    assert set(first_selection) != set(range(24))
    assert output_path.read_bytes().startswith(b"\x89PNG")
