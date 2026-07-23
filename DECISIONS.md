# Reproducibility conventions

This document records shared scientific and implementation choices in Robust
Steering. Exact prompt identities, model configurations, exclusions, and selection
objectives are recorded in benchmark specifications, saved manifests, and the
documentation accompanying each analysis.

## Data separation and controller selection

Direction and dynamics fitting, disturbance estimation, controller selection,
and final evaluation are distinct pipeline stages. Disjoint construction,
calibration, and evaluation prompts are the intended default. Actual separation
is benchmark-specific: HarmBench reuses its small direct-validation pool for
disturbance fitting and selection; MGSM builds its direction from matched
English–Spanish forms and evaluates other input languages; the Llama L-CiteEval
protocol selects on the matched 8K question identities before frozen 16K
evaluation. These arrangements are documented in the corresponding benchmark
protocols rather than represented as universally disjoint splits.

Controller artifacts and selected parameters are frozen for the reported shifted
evaluation. Synthetic controller checks exercise numerical behavior and do not
establish empirical language-model generalization.

## Coordinates and disturbance geometry

H-infinity identification and synthesis use target-preserving orthonormal reduced
coordinates, without whitening, output scaling, disturbance scaling, or depth
weighting. State, control, and terminal penalties are isotropic:
`Q = qI`, `R = rI`, and `Qf = qfI`. Disturbance channels satisfy
`D[k] D[k]ᵀ = covariance(calibration residuals at layer k)`.

These coordinates and costs define the minimax problem. Changing their scale
changes the synthesized controller. Raw-coordinate `gamma_star` can consequently
retain model-scale dependence. Visualization or reporting normalization is
separate from synthesis and does not alter saved controller inputs or gains.

## Decoding and the KV cache

The evaluated language model uses KV caching off by default for Original and
every steering method. Explicit cache-on comparisons use separate evaluation and
result namespaces. Under the activation-hook feedback implementation, cache-on
and cache-off generation can produce different controller trajectories, making
cache state part of the intervention condition.

Each run records the evaluated-model cache setting. Evaluator and judge models
may use their own KV caches because they have no activation feedback policy.

## Baseline parameters

Reported non-H-infinity methods use fixed settings rather than benchmark-specific
parameter sweeps. Source-preserved values are used where available. Some inherited
protocols lacked a recorded final choice; the following explicit settings were
frozen for those comparisons:

| Truthfulness model | Method | Fixed settings |
|---|---|---|
| Gemma-2-2B | ITI | 32 heads, alpha 10 |
| Gemma-2-2B | S-PID | lambda 1; Kp 0.7, Ki 0.01, Kd 0.1 |
| Llama-3-8B | ITI | 32 heads, alpha 10 |
| Llama-3-8B | S-PID | lambda 1; Kp 0.1, Ki 0.1, Kd 0 |
| Llama-3-8B | ODESteer | Layer 19, time 25 |

The ITI and S-PID settings are central choices from preserved source grids. The
Llama ODESteer setting carries the preserved same-model comparison configuration
into truthfulness. These are documented fixed assumptions, not selections made
on final benchmark outcomes. H-infinity cost selection uses the calibration
objective and candidate grid recorded for each model and benchmark.

## Artifact storage

The local collection contains the consolidated experiment results and preserved
remote artifacts. Locally, benchmark caches live under
`benchmarks/<benchmark>/cache/`; tracked specifications, logs, compact results,
tables, and figures remain in their existing repository folders. Large datasets,
model weights, and scientific caches are excluded from Git.

The implementation also supports execution on Lightning: persistent benchmark
artifacts use `~/robust-steering-cache/`, with `LIGHTNING_ARTIFACTS_DIR` as an
explicit override. Model-download and environment caches may use temporary
node-local storage. These runtime locations are separate from the local
preservation collection and do not imply that the original remote machines are
required to reproduce an experiment.
