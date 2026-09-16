# Flow-based trajectory forecasting comparison

This note separates two meanings of *flow* that must not be conflated:

1. a probabilistic normalizing flow that maps a known base density to a conditional future density and uses a change-of-variables likelihood; and
2. an invertible neural map trained directly to regress one observed trajectory representation to one future trajectory representation.

KineRoute NVP, as currently implemented in this repository, is the second kind.

## Comparison

| Method | Flow maps what? | Conditional distribution? | Multiple futures? | Prediction type | Training objective | Domain |
|---|---|---:|---:|---|---|---|
| [FloMo](https://arxiv.org/abs/2103.03614) | A standard-normal latent vector to an entire future trajectory, through conditional rational-quadratic-spline flow modules. The observed trajectory is encoded and conditions the flow. | Yes: models `p(future | observed history)`. | Yes, by drawing multiple latent samples. | Stochastic; supplies tractable sample likelihoods. | Maximum likelihood / negative log-likelihood using change of variables, with noise injection for stable trajectory-density training. | Motion forecasting for pedestrians and vehicles. |
| [Predicting Flow (P-Flow)](https://doi.org/10.1109/LRA.2021.3133862) | A conditionally parameterized normalizing flow represents the probability distribution of future human trajectories. | Yes, according to the publisher-indexed abstract. | Yes; the paper reports multiple plausible trajectories, exact probabilities for predicted samples, and future-position probability maps. | Stochastic/probabilistic. | A likelihood-based normalizing-flow objective is implied by the tractable density model, but the exact loss expression has **not been independently checked here** because the full IEEE text was unavailable. | Human trajectory prediction for robotics/autonomous-driving settings. |
| [FlowChain](https://arxiv.org/abs/2308.08824) | A chain of conditional continuously-indexed flows transforms a Gaussian density at the latest observation into successive per-future-step spatial densities. | Yes: estimates `p(x_(t+n) | observed history)` at every future step. | Yes, by sampling the Gaussian/base-flow chain. | Stochastic density prediction; also supports analytic density maps and fast online density updates. | Each conditional flow is trained by negative log-likelihood, using its analytic/approximated change-of-variables density. | Human trajectory prediction. |
| KineRoute NVP (this repository) | One normalized 60-D chart of the observed 30-point trajectory directly to one normalized 60-D chart of the future 30-point trajectory. Routed affine couplings split the vector into anchor and two 29-D motion groups. | **No.** There is no base distribution, conditional density, log-determinant accumulation, or `p(future | history)`. | **No.** A fixed history produces one fixed future. | Deterministic point forecast with an analytic inverse used as a reverse diagnostic/regularizer. | Supervised ADE plus optional position-MSE, velocity, acceleration, turn, feasibility, and reverse-ADE terms. | Vessel trajectory forecasting on EnvShip. |

## What KineRoute's invertibility does and does not mean

KineRoute's coupling stack is mathematically bijective as a map between equal-dimensional history and future chart vectors, and its inverse can be evaluated numerically. In the current system, however, it does not transform samples from a base probability distribution, compute Jacobian log-determinants, maximize data likelihood, or expose calibrated probabilities. Calling the implementation a deterministic, routed, invertible trajectory regressor is therefore precise; calling it a probabilistic normalizing-flow density estimator would be incorrect.

The defensible experimental question is narrower: after controlling preprocessing, objective, capacity, and split, does this routed bijective architecture improve deterministic vessel point-forecast ADE/FDE, and do routing or invertibility contribute? The planned standard-RealNVP and non-invertible routed controls can answer those architecture questions. They cannot by themselves support a claim of novelty or superiority over probabilistic multi-future methods such as FloMo, P-Flow, or FlowChain.

## Evidence and limits

- FloMo details above come from the primary arXiv paper, especially its problem formulation and method sections.
- FlowChain details above come from the primary arXiv paper, especially Sections 3.2--3.4.
- P-Flow bibliographic metadata and method-level claims come from the IEEE DOI record and its indexed abstract. Its exact loss equation remains deliberately unclaimed until the full paper is inspected.
- KineRoute details were checked against `models/kineroute_nvp.py`, `models/coupling.py`, and the unified trainer in this repository.

