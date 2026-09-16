# Experiment Artifact Contract

Current artifact schema version: 2. Version 2 adds an exact runtime-source
fingerprint so dirty-worktree runs cannot be silently mixed.

Every training method uses `RunArtifacts` from
`src/kineroute_nvp/training/artifacts.py`. New models and ablations should use
the same component instead of creating result folders or checkpoint formats in
their own runners.

## Run layout

Each invocation creates one immutable directory:

```text
results/<timestamp>__<experiment_name>__<run_name>/
├── _results.json
├── command.txt
├── environment.json
├── resolved_config.yaml
├── checkpoints/
│   ├── best.pt
│   ├── epoch_0010.pt
│   ├── epoch_0020.pt
│   └── final.pt
├── evaluation/
│   ├── metrics.csv
│   ├── metrics.json
│   ├── provenance.json (when produced by standalone evaluation)
│   └── test_predictions_summary.json
├── figures/
│   ├── val_ade_vs_epoch.png
│   ├── val_fde_vs_epoch.png
│   └── trajectory_examples.png
└── logs/
    ├── metrics.jsonl
    └── train.log
```

The validation plots and `_results.json` are created before the first epoch.
They are refreshed after every completed epoch, so a running, interrupted, or
failed run still has inspectable progress artifacts.

Trajectory examples use a seeded random selection of exactly
`evaluation.save_prediction_examples` test samples (or the complete test split
when it is smaller) and label each panel with its dataset `sample_id`.
Regenerate 20 examples from a saved best checkpoint with:

```bash
python scripts/visualize_trajectories_from_pt.py \
  --pt results/<run-folder>/checkpoints/best.pt \
  --n 20 --random --seed 42 --grid \
  --output-dir results/<run-folder>/figures
```

Training writes the selected sample IDs to
`evaluation/test_predictions_summary.json` so the examples can be traced back
to the fixed test split. The standalone visualization command writes its own
selection manifest alongside its figures.

## Checkpoints

The default and paper configuration interval is 10 epochs. Periodic
checkpoints are named `epoch_NNNN.pt`. `best.pt` is selected by validation ADE,
and `final.pt` records the terminal trained state before the best checkpoint is
restored for final evaluation.

Hyperparameter and architecture-search configs can set
`evaluation.evaluate_test: false`. In that mode the runner never constructs or
reads the test dataset, writes `validation_metrics.json`, and records
`test_evaluated: false` with null test metrics. After selection is frozen,
`scripts/evaluate.py` loads the chosen `best.pt` for the one final test
evaluation; it never retrains.

Every checkpoint contains:

- model state
- optimizer state
- scheduler state
- epoch
- checkpoint-selection metadata
- Python, NumPy, Torch, and CUDA RNG states when available
- fully resolved configuration
- Git/runtime-source/dataset provenance captured by the run
- artifact schema version

The payload contains the state required for resumption, but the current runner
does not expose a `--resume` workflow; interrupted runs are restarted into a
new immutable result directory. Loading only `checkpoint["model_state"]` is
sufficient for evaluation.

## Results lifecycle

`_results.json` is written atomically and has a `status` of `running`,
`completed`, `interrupted`, or `failed`. It includes the complete per-epoch
history, last completed epoch, configuration and command provenance, artifact
paths, best validation metrics, whether test evaluation occurred, final test
metrics when permitted, immutable manifest identity for frozen runs, and
checkpoint paths.

It also stores `runtime_source_identity`, a SHA-256 fingerprint over the
importable `src/**/*.py` implementation and `scripts/train.py`. A commit hash
alone cannot identify uncommitted code in a dirty research worktree. Strict
ablation and multi-seed aggregation rejects mixed runtime-source fingerprints.

## Running the paper benchmark

Run every current coordinate-only baseline and Kine-Real-NVP:

```bash
bash experiments/baselines/run_all.sh
```

Command-line overrides are forwarded to every paper configuration. For
example, on a host without CUDA or multiprocessing support:

```bash
bash experiments/baselines/run_all.sh --device cpu --num-workers 0
```

Run one method:

```bash
python scripts/train.py --config experiments/baselines/kine_real_nvp_paper.yaml
python scripts/train_baseline.py --config experiments/baselines/gru_coords_paper.yaml
```

Compare explicit result folders in one two-panel validation figure:

```bash
python scripts/plot_baseline_curves.py \
  20260915_160446__baselines__kine_real_nvp_paper \
  20260915_144806__baselines__bigru_coords_paper \
  --output results/validation_curves_combined.png
```

Folder names are resolved below `results/`; complete paths also work. The
script reports the stored seed, validation split name, sample count, and warns
when the supplied runs do not describe a compatible validation dataset.

Do not reuse a run directory. A new timestamped directory is created for every
invocation.

## Adding a new method

Create one `RunArtifacts` instance at the start of the run. Call
`record_epoch()` once after each validation pass, save the best checkpoint when
the selection metric improves, call `save_periodic_checkpoint()` after every
epoch, and finish with `final.pt` plus `complete()`. On exceptions, call
`fail()` before re-raising. This keeps future methods and ablations compatible
with the same reporting and aggregation tools.
