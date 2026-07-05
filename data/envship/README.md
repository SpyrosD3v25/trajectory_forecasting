# EnvShip-Bench

EnvShip-Bench is a benchmark for short-term vessel trajectory prediction built from raw AIS data released by the Danish Maritime Authority (DMA).

This Hugging Face release currently provides two source branches under a shared layout:

- `DMA/`
- `NOAA/`

The DMA branch is organized under:

- `DMA/benchmark/core/`
- `DMA/benchmark/full/`
- `DMA/mini_benchmark/ship_core_lite/`
- `DMA/mini_benchmark/clean_ship_core_lite_v1/`

The NOAA branch is organized under:

- `NOAA/benchmark/core/`
- `NOAA/benchmark/full/`
- `NOAA/mini_bench/clean_ship_core_lite_v1/`

The layout keeps each source in its own top-level directory so new branches can be added without changing the public repository structure.

## Forecasting Protocol

All benchmark samples follow the same protocol:

- Observation horizon: 10 minutes
- Prediction horizon: 10 minutes
- Sampling interval: 20 seconds
- History length: 30 points
- Future length: 30 points

## Included Data

### 1. `DMA/benchmark/core`

The main large-scale benchmark release with strict quality control for standardized vessel trajectory prediction.

### 2. `DMA/benchmark/full`

A more inclusive release that retains additional valid windows beyond the strict core subset.

### 3. `DMA/mini_benchmark/ship_core_lite`

A lightweight representative mini benchmark for quick experimentation and method prototyping.

### 4. `DMA/mini_benchmark/clean_ship_core_lite_v1`

A quality-first compact benchmark with stricter filtering and controlled motion profiles.

This subset also includes:

- `environment_v1`
- `environment_v2`
- `social_env_v1`

These packages support environment-aware and interaction-aware vessel forecasting on top of the same compact split.

