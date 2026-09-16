# Frozen Compact EnvShip Protocol

All compact ablations use one frozen train/validation/test split. The five
training seeds are `0, 1, 2, 3, 4`; they change stochastic training behavior
only. They never select, shuffle between, or regenerate dataset splits.

The canonical manifest is
`experiments/compact/split_manifest.json`. It stores every ordered,
source-qualified sample ID, the sample counts and SHA-256 fingerprint for each
split, source shard identities, and the data protocol. Source qualification is
required because one raw `sample_id` occurs in both the DMA and NOAA training
sources.

Frozen counts:

| Split | Samples |
|---|---:|
| Train | 29,635 |
| Validation | 3,667 |
| Test | 3,262 |

The protocol is exactly 30 observed points followed by 30 future points, at a
20-second interval. The source data card states that positions use a local
planar frame centered on the final history point, with x east, y north, and
units of metres. The manifest records both this statement and the SHA-256 of
the local data-card copy. Therefore ADE and FDE computed directly from these
coordinates are in metres. This conclusion is tied to the frozen source
identity; a different dataset must provide its own unit evidence.

For ordinary forecasters, preprocessing is defined from history only. Let
`o = p_30` and let `theta` be the heading of the final non-stationary observed
step (or zero when all observed steps are stationary). Heading alignment is
`q_t = R(-theta)(p_t - o)`, and the same `o, theta` encode the future. The
displacement representation uses `d_1 = q_1` and
`d_t = q_t - q_(t-1)` thereafter; cumulative summation followed by the inverse
rotation and translation reconstructs metric positions.

With `loss.normalize: true`, the MSE objective is computed in normalized model
representation space. The ADE objective, model selection ADE, and all reported
ADE/FDE metrics are computed only after decoding predictions back to metric
coordinates. Thus reported ADE/FDE are always metres even when the training
representation is aligned or displaced.

Manifest-backed preparation and training fail if source fingerprints, ordered
split membership, split counts, shapes, configured horizons, or sampling
interval differ. Processed metadata additionally records a semantic SHA-256
fingerprint for each split over its ordered IDs and all four arrays (history,
future, history chart, and future chart). Preflight recomputes it, so changed
coordinates cannot pass merely because IDs and shapes stayed fixed. Legacy
processed roots require one explicit regeneration to acquire these fingerprints;
experiment launchers never repair data automatically.

Regenerate the manifest only as an explicit dataset-version change:

```bash
python scripts/freeze_compact_split.py
```

Normal experiment runs must reuse the checked-in manifest and must not run that
command. Changing a training seed does not change `data.split_manifest`,
`data.train_split`, `data.val_split`, or `data.test_split`.
