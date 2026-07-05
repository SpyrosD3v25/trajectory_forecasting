# KineRoute-NVP

KineRoute-NVP is a focused research implementation for deterministic short-term vessel trajectory forecasting on EnvShip-Bench. The repository stays intentionally narrow: one model, one experiment interface, one artifact tree.

The model learns a supervised invertible map from 30 observed vessel positions to 30 future positions using a routed RealNVP architecture over a heading-aligned displacement chart. Evaluation is reported with Average Displacement Error (ADE) and Final Displacement Error (FDE) in meters.

