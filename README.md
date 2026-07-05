# KineRoute-NVP

KineRoute-NVP is a focused research implementation for deterministic short-term vessel trajectory forecasting on EnvShip-Bench. The repository stays intentionally narrow: one model, one experiment interface, one artifact tree.

The model learns a supervised invertible map from 30 observed vessel positions to 30 future positions using a routed RealNVP architecture over a heading-aligned displacement chart. Evaluation is reported with Average Displacement Error (ADE) and Final Displacement Error (FDE) in meters.

## Dataset

Experiments use the official compact paper subsets from:

`mark000071/EnvShip-Bench_An_Environment-Enhanced_Benchmark_for_Short-Term_Vessel_Trajectory_Prediction`

Local aliases:

- `data/envship/paper_dma_clean_ship_core_lite_v1`
- `data/envship/paper_noaa_clean_ship_core_lite_v1`

Combined split sizes:

- train: `29635`
- val: `3667`
- test: `3262`

Download and verify:

```bash
python scripts/download_envship.py --datasets dma_clean noaa_clean
python scripts/verify_download.py --dataset dma_clean
python scripts/verify_download.py --dataset noaa_clean
```

## Experiments

Every run is defined by one YAML under `experiments/`. One YAML corresponds to one run.

```bash
python scripts/train.py --config experiments/baselines/kine_real_nvp_paper.yaml
```

Batch execution stays reproducible through checked-in launchers:

```bash
bash experiments/baselines/run_all.sh
```

