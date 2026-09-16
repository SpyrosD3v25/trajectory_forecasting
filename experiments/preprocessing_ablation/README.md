# Preprocessing Ablation

This directory defines the eight Section 4 attribution runs for GRU and LSTM:
raw + MSE, heading-aligned + MSE, heading-aligned displacement + MSE, and
heading-aligned displacement + ADE.

Every configuration uses:

- the same frozen manifest at `experiments/compact/split_manifest.json`;
- the same train/validation/test membership;
- the same 30 -> 30, 20-second protocol;
- hidden size 128 and two recurrent layers;
- seed 0 for the initial ablation comparison;
- CUDA for actual training;
- validation-only final evaluation during selection (`evaluate_test: false`).

The seed affects training stochasticity only. Override it without changing the
split, for example `--seed 1`.

Preflight the processed data without training:

```bash
python scripts/preflight_compact_data.py
```

If this fails, inspect the drift first. Regeneration is a separate explicit
operator action with `python scripts/preflight_compact_data.py --repair`; the
experiment launcher never repairs or mutates its inputs.

Run all eight GPU experiments when a CUDA device is available:

```bash
bash experiments/preprocessing_ablation/run_all.sh
```

Additional CLI arguments are forwarded to each run. After completion, produce
the Section 4 table with:

```bash
python scripts/aggregate_preprocessing_ablation.py
```

## Predeclared interpretation and selection rule

The three attribution contrasts are evaluated separately within GRU and LSTM:

1. alignment: `raw + MSE` versus `heading_aligned + MSE`;
2. displacement: `heading_aligned + MSE` versus
   `heading_aligned_displacement + MSE`;
3. objective: `heading_aligned_displacement + MSE` versus
   `heading_aligned_displacement + ADE`.

No conclusion is drawn from an untested cross-factor cell (for example,
`raw + ADE`). The shared preprocessing/objective protocol for subsequent model
families is selected by the lowest arithmetic mean validation ADE across the
GRU and LSTM runs for each of the four tested protocols. Test ADE/FDE is never
used for that choice. Exact ties are broken by lower mean validation FDE, then
by the simpler protocol in the order raw, heading-aligned, aligned displacement
+ MSE, aligned displacement + ADE. This rule was recorded before the remaining
matrix results were inspected.

The test split is not opened by these eight official runs. After the shared
protocol is selected, test evaluation is performed only for the subsequently
selected final candidates. Earlier exploratory artifacts that contain test
metrics are ineligible for this table.
