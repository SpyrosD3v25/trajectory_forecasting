# Experiment 1

Research question: can a single routed RealNVP map 30 observed EnvShip vessel points to the next 30 points while retaining an invertible reverse diagnostic?

The model is KineRoute-NVP only. The canonical config uses a heading-aligned
Cartesian displacement chart with routed affine-coupling blocks over the
2-D anchor, 29 forward-displacement values, and 29 lateral-displacement values.
It uses ADE as the supervised objective; optional velocity, acceleration,
turn, feasibility, and reverse terms are configurable but are all zero in the
canonical paper config. These terms describe trajectory-motion structure, not
full hydrodynamics, currents, or engine physics.

Metrics: ADE, FDE, reverse ADE/FDE, velocity error, acceleration error, turn-rate error, infeasible acceleration percentage, infeasible turn-rate percentage.

`kineroute_paper.yaml` is the canonical, frozen-manifest experiment and is the
only configuration launched by `run_all.sh`. `kineroute_v1.yaml` is retained as
a historical pre-manifest record; it uses a different data source/protocol and
must not be mixed into current tables or launched as part of the canonical
experiment.

Outputs: one immutable `results/<timestamp>__experiment_1__kineroute_paper/`
directory with `_results.json`, resolved config, checkpoints, logs, figures,
and evaluation artifacts.
