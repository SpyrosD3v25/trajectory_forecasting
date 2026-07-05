# Experiment 1

Research question: can a single routed RealNVP map 30 observed EnvShip vessel points to the next 30 points while retaining an invertible reverse diagnostic?

The model is KineRoute-NVP only. It uses a safe polar kinematic chart, routed affine-coupling blocks over anchor, speed, and steering channels, ADE as the main supervised objective, and velocity/acceleration/turn/feasibility regularizers from AIS-derived kinematics. These terms describe trajectory-motion structure, not full hydrodynamics, currents, or engine physics.

Metrics: ADE, FDE, reverse ADE/FDE, velocity error, acceleration error, turn-rate error, infeasible acceleration percentage, infeasible turn-rate percentage.

Outputs: one immutable `results/<timestamp>__experiment_1__kineroute_v1/` directory with `_results.json`, resolved config, checkpoints, logs, figures, and evaluation artifacts.
