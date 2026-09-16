# KineRoute NVP Implementation TODO

This tracker is based on the older prompt, updated against the current repo state.

Checkbox policy:

- `[x]` means the item is genuinely implemented, verified, or executed exactly as written.
- `[ ]` means it is missing, unverified, blocked by a dependency, or still requires a real run.
- Setup and result-dependent experiment items are kept separate; a config is not an executed experiment.
- `PARTIAL:` means something exists, but not enough to count as the prompt item.

Current project decisions:

- Use one common runner unless there is a clear technical reason not to; any exception must be written down.
- Expose preprocessing modes explicitly in YAML: `raw`, `heading_aligned`, `heading_aligned_displacement`.
- Use one frozen compact train/validation/test split shared by all models and all five training seeds.
- Create configs and launcher commands immediately, but do not run long experiments automatically.
- Keep the current baselines and add the missing ones.
- Parameter matching targets +/-5% trainable parameters where feasible and permits up to +/-10% only when architecture discreteness prevents a closer match; exact counts and errors must always be reported.
- Canonical paper/project name: KineRoute NVP.

## 1. Core Modules

- [x] **Implement shared trajectory preprocessing module**
  - Input raw `[30, 2]` trajectories.
  - Support explicit modes: `raw`, `heading_aligned`, `heading_aligned_displacement`.
  - Implement inverse transform back to metric coordinates.
  - Acceptance: `decode(encode(x)) ~= x` for displacement/alignment pipeline.
  - Shared `TrajectoryPreprocessor` supports explicit modes and inverse metric-coordinate decoding.
  - Aligned modes use `origin = history[-1]` and the final valid history-motion heading; future transforms reuse those history-only parameters.
  - Displacement convention is `d[0] = q[0]`, `d[t] = q[t] - q[t-1]`, so cumulative summation reconstructs `q` exactly up to float32 rounding.

- [x] **Implement shared ADE/FDE evaluator**
  - Shared ADE/FDE functions exist in `src/kineroute_nvp/losses/trajectory_physics.py`.
  - Used by KineRoute and baseline evaluation paths.

- [x] **Implement MSE trajectory loss**
  - Shared MSE trajectory loss exists and baseline trainer supports `loss.name: mse`.

- [x] **Implement ADE trajectory loss**
  - `ade()` exists and baseline trainer supports `loss.name: ade`.

- [x] **Implement parameter-count utility / metadata field**
  - Shared trainable parameter-count utility exists and is wired into baseline and KineRoute metadata.

- [x] **Implement deterministic seed utility**
  - `set_seed()` seeds Python, NumPy, PyTorch CPU/GPU and is wired from YAML/CLI.

- [x] **Implement Dead Reckoning model**
  - Estimate velocity from last observed points.
  - Extrapolate constant velocity for 30 steps.
  - No trainable parameters.

- [x] **Implement configurable GRU forecaster**
  - Configurable hidden size/layers; predicts all 30 future positions.
  - Bidirectional variants concatenate the terminal forward/backward hidden states rather than using an incomplete backward output at the final time index.

- [x] **Implement configurable LSTM forecaster**
  - Configurable hidden size/layers; predicts all 30 future positions.
  - Bidirectional variants use both terminal direction states.

- [x] **Implement non-invertible MLP forecaster**
  - Flatten history to future trajectory.
  - Configurable hidden width/depth.

- [x] **Implement residual MLP forecaster**
  - Same I/O as MLP.
  - Configurable residual blocks.

- [x] **Implement TCN forecaster**
  - 30-step history input.
  - 30-step future output.
  - Same preprocessing/loss interface.

- [x] **Implement Transformer-NAR forecaster**
  - Non-autoregressive output of all 30 future steps.
  - Same preprocessing/loss interface.

- [x] **Verify current KineRoute/RealNVP implementation**
  - Forward/inverse path exists.
  - Routed coupling order exists.
  - Numerical invertibility test exists.
  - Shared preprocessing tests cover the heading-aligned displacement round trip.

- [x] **Implement ordinary/non-routed RealNVP**
  - Same approximate parameter budget as KineRoute.
  - Standard arbitrary partition instead of physical routing.
  - Needed to isolate routing itself.

- [x] **Implement non-invertible routed control**
  - Preserve `A/S/T` routing structure.
  - Remove invertibility constraint.
  - Needed to isolate invertibility itself.

## 2. Training / Experiment System

- [x] **Create one common model interface**
  - `forward(history) -> future`.
  - All deterministic baselines conform to it.
  - Deterministic forecasters share the `history -> future` interface; chart-space invertible models are adapted inside the unified runner.

- [x] **Create one common trainer**
  - Model-independent.
  - Supports MSE or ADE.
  - Validation every epoch.
  - Early/best checkpoint by validation ADE.
  - `scripts/train.py` drives all Section 1 trainable models plus parameter-free Dead Reckoning; `scripts/train_baseline.py` is now a compatibility wrapper.

- [x] **Create one common evaluation script**
  - Loads checkpoint.
  - Runs test set once.
  - Outputs ADE/FDE/params.
  - `scripts/evaluate.py` loads a checkpoint or result directory and evaluates one split.

- [x] **Make preprocessing configurable from YAML**
  - Explicit modes: `raw`, `heading_aligned`, `heading_aligned_displacement`.
  - Unified config maps preprocessing mode to the concrete heading-alignment/chart settings while preserving old config compatibility.

- [x] **Make loss configurable from YAML**
  - Unified YAML supports `mse`, `ade`, and `kineroute` losses.

- [x] **Make model configurable from YAML**
  - Required models: `dead_reckoning`, `gru`, `lstm`, `bigru`, `bilstm`, `seq2seq`, `mlp`, `resnet`, `tcn`, `transformer_nar`, `realnvp`, `routed_realnvp`.
  - Unified model factory builds all Section 1 models from YAML/CLI configuration.

- [x] **Make architecture hyperparameters configurable**
  - Hidden size, layers, RealNVP blocks, learning rate, epochs, batch size, seed.
  - Unified parser exposes model-family hyperparameters with CLI-over-YAML precedence.

- [x] **Standardize result folder output**
  - Config, params where supported, best epoch, validation ADE/FDE, test ADE/FDE, seed, checkpoints, runtime/log history.
  - Shared artifact contract exists under `src/kineroute_nvp/training/artifacts.py`.

- [x] **Create experiment aggregator**
  - Scan result folders.
  - Produce one CSV with one row per run.
  - `scripts/aggregate_results.py`.

- [x] **Create multi-seed aggregator**
  - Group identical configs.
  - Compute ADE/FDE mean and std.
  - `scripts/aggregate_multiseed.py`.

## 3. Verification Before Experiments

- [x] **Freeze one compact EnvShip train/validation/test split**
  - `experiments/compact/split_manifest.json` stores every ordered, source-qualified sample ID.
  - Frozen counts: 29,635 train / 3,667 validation / 3,262 test.
  - Per-source and per-split SHA-256 fingerprints make source or membership drift fail loudly.
  - Train, validation, and test are verified disjoint.
  - Repeated preparation is verified to preserve the same split and reject changed source data.

- [x] **Verify coordinate units**
  - The frozen manifest records the hashed EnvShip data-card evidence for the anchor-relative local planar frame.
  - `hist_x/y` and `fut_x/y` are documented as metres, so ADE/FDE over those coordinates are metres.
  - `docs/compact-split-protocol.md` documents the evidence and scope of this conclusion.

- [x] **Verify 30 -> 30 sampling**
  - All frozen source coordinate arrays are checked as exactly 30 history and 30 future points.
  - Processed datasets enforce `[N, 30, 2]` history and future shapes.
  - Config and processed metadata must preserve `history_steps=30`, `future_steps=30`, and `dt_seconds=20.0`.

- [x] **Verify heading alignment has no leakage**
  - Tests prove origin, angle, and encoded history are unchanged when only the future changes.
  - Alignment is fitted from observed history only, including stationary edge cases.

- [x] **Verify displacement transform has no leakage**
  - Tests prove the ordinary-forecaster input depends only on history.
  - Future encoding uses the history-fitted transform without changing model input.

- [x] **Verify normalization and feasibility statistics use train split only**
  - Tests change validation/test coordinates while holding train fixed.
  - Coordinate normalization, chart normalization, and feasibility thresholds remain identical.

- [x] **Unit-test inverse preprocessing**
  - `raw`, `heading_aligned`, and `heading_aligned_displacement` round-trip multiple curved, stationary, and near-stationary trajectories.
  - Future encoding/decoding is verified with a transform fitted from history.

- [x] **Smoke-test every Section 1 model family through the shared system**
  - Dead Reckoning, GRU, LSTM, MLP, ResNet, TCN, Transformer-NAR, standard RealNVP, KineRoute NVP, and non-invertible routed control are covered.
  - Trainable models run preprocessing, forward, configured loss, backward, optimizer step, common checkpoint save/reload, and common evaluation on CPU-tiny data.
  - Dead Reckoning follows the same artifact/evaluation path without optimizer or backward.

- [x] **Verify RealNVP invertibility numerically**
  - Standard RealNVP and routed KineRoute NVP pass strict `inverse(forward(x)) ~= x` checks after checkpoint reload.
  - The non-invertible routed control is explicitly excluded and has no `.inverse()` path.

- [x] **Verify all model paths use the identical metric evaluator**
  - Runtime regression coverage proves every Section 1 model family reaches the shared `summarize_predictions` ADE/FDE path.

- [x] **Verify fixed split with varying training seeds**
  - Seeds `0, 1, 2, 3, 4` are recorded in the manifest as training seeds only.
  - Config tests prove seed 0 and seed 1 resolve to identical manifest and split settings while retaining different seed values.

## 4. Preprocessing Ablation Experiment Setup

Configs and launcher commands are ready. The host RTX 2050 is CUDA-visible to the project environment. The launcher and configs are verification-only and never repair missing/drifted processed data implicitly.

- [x] **Set up GRU raw + MSE**
  - Baseline `G0`.

- [x] **Set up LSTM raw + MSE**
  - Baseline `L0`.

- [x] **Set up GRU heading-aligned + MSE**
  - Compare directly against `G0`.

- [x] **Set up LSTM heading-aligned + MSE**
  - Compare directly against `L0`.

- [x] **Set up GRU heading-aligned + displacement + MSE**
  - Isolates displacement effect.

- [x] **Set up LSTM heading-aligned + displacement + MSE**

- [x] **Set up GRU heading-aligned + displacement + ADE**
  - Isolates objective effect.

- [x] **Set up LSTM heading-aligned + displacement + ADE**

- [x] **Generate preprocessing-ablation table script**
  - Columns: `model / alignment / displacement / loss / params / ADE / FDE`.
  - `scripts/aggregate_preprocessing_ablation.py` emits validation ADE/FDE per selection run; official Section 4 runs must not open the test split.
  - Final mode requires all eight exact cells and rejects mixed seeds, manifests, protocols, data roots, split counts, controlled hyperparameters, duplicate cells, failed/incomplete runs, and non-finite metrics.
  - `experiments/preprocessing_ablation/run_all.sh` performs read-only preflight, then launches all eight CUDA configs.

- [ ] **Run all eight preprocessing ablations and select preprocessing/objective by validation ADE**
  - Command: `bash experiments/preprocessing_ablation/run_all.sh`.
  - Then run: `python scripts/aggregate_preprocessing_ablation.py`.
  - Exploratory CUDA attempts exposed and fixed a headless Matplotlib/DataLoader worker crash, legacy per-candidate test evaluation, a global-vs-per-feature normalization confound, and missing dirty-worktree source provenance. Those artifacts are excluded. The official matrix will run validation-only from one verified runtime-source fingerprint.

## 5. Strong Baseline Experiment Setup

- [ ] **Set up Dead Reckoning**

- [ ] **Set up MLP with aligned displacement + ADE**

- [ ] **Set up ResNet/MLP with aligned displacement + ADE**

- [ ] **Set up TCN with aligned displacement + ADE**

- [ ] **Set up Transformer-NAR with aligned displacement + ADE**

- [x] **Keep current baseline configs**
  - Current setup includes GRU, BiGRU, LSTM, BiLSTM, Seq2Seq, and KineRoute NVP paper configs.

- [ ] **Select strongest deterministic baseline using validation ADE**
  - Do not use test ADE for model selection.
  - Requires completed runs later.

## 6. RealNVP Architecture Ablation Setup

- [ ] **Set up blocks=1, hidden=64**
- [ ] **Set up blocks=1, hidden=128**
- [ ] **Set up blocks=1, hidden=256**
- [ ] **Set up blocks=2, hidden=64**
- [ ] **Set up blocks=2, hidden=128**
- [ ] **Set up blocks=2, hidden=256**
- [ ] **Set up blocks=3, hidden=64**
- [ ] **Set up blocks=3, hidden=128**
- [ ] **Set up blocks=3, hidden=256**
- [ ] **Set up blocks=4, hidden=64**
- [ ] **Set up blocks=4, hidden=128**
- [ ] **Set up blocks=4, hidden=256**
- [ ] **Set up blocks=6, hidden=64**
- [ ] **Set up blocks=6, hidden=128**
- [ ] **Set up blocks=6, hidden=256**

- [ ] **Aggregate the 15 RealNVP setups**
  - Table: `blocks / hidden / params / val ADE / test ADE / FDE`.

- [ ] **Select best RealNVP architecture by validation ADE**
  - Requires completed runs later.

## 7. Capacity Controls

- [ ] **Measure strongest baseline parameter count**
  - Requires completed baseline selection later.

- [ ] **Create parameter-matched RealNVP**
  - Target +/-5%; allow up to +/-10% only if discrete widths prevent a closer match.

- [ ] **Set up parameter-matched RealNVP**

- [ ] **Create parameter-matched MLP/ResNet**
  - Target +/-5%; allow up to +/-10% only if discrete widths prevent a closer match.

- [ ] **Set up parameter-matched MLP/ResNet**

- [ ] **Produce capacity-control comparison script/table**
  - `baseline vs ResNet vs RealNVP`.
  - Include exact parameter counts.

## 8. Routing / Invertibility Ablations

- [ ] **Set up standard RealNVP**
  - Same preprocessing.
  - Same loss.
  - Same approximate parameter budget.
  - No physical routing.

- [x] **Set up routed KineRoute NVP**
  - Current KineRoute NVP config exists.

- [ ] **Compare routed vs standard RealNVP**
  - Answers whether routing matters.

- [ ] **Set up non-invertible routed model**
  - Same `A/S/T` grouping.

- [ ] **Compare invertible routed vs non-invertible routed**
  - Answers whether invertibility matters.

## 9. Multi-Seed Verification Setup

- [ ] **Freeze final configurations**
  - No hyperparameter changes after this point.

- [ ] **Set up training seed 0 on the one frozen split for every final model**
- [ ] **Set up training seed 1 on the one frozen split for every final model**
- [ ] **Set up training seed 2 on the one frozen split for every final model**
- [ ] **Set up training seed 3 on the one frozen split for every final model**
- [ ] **Set up training seed 4 on the one frozen split for every final model**

Final models should include at least:

- GRU strongest
- LSTM strongest
- TCN
- Transformer-NAR
- MLP/ResNet matched
- standard RealNVP
- KineRoute NVP

- [ ] **Compute mean +/- std ADE**
- [ ] **Compute mean +/- std FDE**
- [ ] **Compute std across seeds separately**
- [ ] **Determine whether KineRoute NVP is actually more stable**

## 10. Literature Work

- [x] **Read FloMo**
  - Extract input, output, flow variable, conditioning, objective, deterministic/stochastic, role of invertibility.

- [ ] **Read P-Flow**
  - Extract the same fields.
  - PARTIAL: publisher metadata and the indexed abstract establish a conditionally parameterized probabilistic normalizing flow, multiple futures, exact sample probabilities, and probability maps. The full IEEE text was unavailable, so the exact loss equation is not marked verified.

- [x] **Read FlowChain**
  - Extract the same fields.

- [x] **Create `literature/flow_comparison.md`**
  - Rows: FloMo, P-Flow, FlowChain, KineRoute NVP.
  - Columns: `flow maps what / conditional distribution / multi-future / deterministic / objective / domain`.

- [x] **Write one paragraph defining exactly how KineRoute NVP differs**
  - No novelty claim beyond what the experiments support.

## 11. Final Analysis Artifacts

- [ ] **Generate preprocessing attribution table**
- [ ] **Generate RealNVP architecture table**
- [ ] **Generate parameter-matched comparison table**
- [ ] **Generate routing/invertibility ablation table**
- [ ] **Generate 5-seed mean+/-std table**
- [ ] **Generate ADE/FDE bar/point plot**
- [ ] **Generate gain-attribution plot**
  - Raw -> alignment -> displacement -> ADE -> architecture.

- [ ] **Write one compact conclusion**
  - Decide which statement is actually supported:
    - preprocessing caused gain;
    - capacity caused gain;
    - RealNVP helped;
    - routing helped;
    - invertibility helped;
    - combination helped.

## 12. Only After Compact Dataset Succeeds

- [ ] **Freeze final architecture/hyperparameters**
- [ ] **Set up full EnvShip Track A run**
- [ ] **Set up strongest baselines on Track A**
- [ ] **Set up KineRoute NVP on Track A**
- [ ] **Set up key models across 5 seeds if feasible**
- [ ] **Set up cross-region evaluation**
- [ ] **Produce final full-dataset comparison**

## First Executable Batch

Do this first, before TCN/Transformer/extra RealNVP variants:

- [x] Finish shared preprocessing modes and inverse tests.
- [x] Finish shared evaluator/test coverage.
- [x] Ensure GRU and LSTM use the common runner/interface.
- [x] Implement Dead Reckoning.
- [x] Unify trainer/config path unless a clear blocker is documented.
- [x] Add one frozen compact split manifest shared by all five training seeds.
- [x] Add the 8 preprocessing-ablation configs and launcher commands.

This first block answers the main attribution question: whether the gain is coming from preprocessing before adding more architecture complexity.

## Resolved analysis decisions

- The core deterministic table contains Dead Reckoning, GRU, LSTM, MLP,
  residual MLP, TCN, and Transformer-NAR. Seq2Seq, BiGRU, and BiLSTM remain
  useful additional/appendix baselines so they do not obscure the required
  attribution chain.
- Dead Reckoning uses the final two observed points (`velocity_points: 2`), a
  predeclared constant-velocity rule with no tuned window length.
- Capacity matching uses the target appropriate to each controlled question:
  standard RealNVP is matched to the strongest deterministic baseline, while
  the MLP/ResNet control is matched to the selected KineRoute/RealNVP model.
  Exact counts and percent errors are reported in both cases.
