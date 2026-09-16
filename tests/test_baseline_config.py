from kineroute_nvp.utils.baseline_config import args_to_config, parse_args


def test_baseline_yaml_parsing() -> None:
    args = parse_args(["--config", "experiments/baselines/gru_coords_paper.yaml"])
    config = args_to_config(args)
    assert args.experiment_name == "baselines"
    assert config["run_name"] == "gru_coords_paper"
    assert config["model"]["name"] == "gru"
    assert config["training"]["checkpoint_every_epochs"] == 10
    assert config["data"]["raw_roots"] == [
        "data/envship/paper_dma_clean_ship_core_lite_v1",
        "data/envship/paper_noaa_clean_ship_core_lite_v1",
    ]
