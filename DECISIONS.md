# Decisions

Use this file only for choices that must stay consistent across multiple analyses or outputs.

Do not put unit-specific file paths, temporary subsets, cache names, panel mappings, or one-off thresholds here. Put those details in the owning compact unit README.

## Split Isolation

- Decision: Use disjoint fit, calibration, and test prompt sets. Fit semantic directions and nominal dynamics on the fit split, estimate disturbance geometry on the calibration split, and evaluate all reported steering reliability on the untouched test split.
- Why: Robust steerability is intended to predict held-out intervention reliability rather than describe the same prompts used to construct the representation and controller.
- Use this when: Constructing semantic targets, identifying representation dynamics, estimating disturbance channels, tuning controllers, or reporting predictive performance.
- Do not use this for: Self-contained synthetic controller checks that do not support an empirical language-model claim.
- Notes: Record the exact prompt identifiers and any exclusions in `data/README.md` and the owning compact-unit README.

## Cross-Model Coordinates

- Decision: Identify and synthesize H∞ in the target-preserving orthonormal reduced coordinates without whitening, output scaling, disturbance scaling, or depth weighting. Use `Q=qI`, `R=rI`, `Qf=qfI`, and construct each `D[k]` so `D[k]D[k]ᵀ` equals the empirical calibration-residual covariance.
- Why: These are the coordinates and cost definitions supplied to the minimax problem; silently rescaling them changes the controller rather than merely changing how its result is reported.
- Use this when: Building `A/B/D/Q/R/Qf`, synthesizing H∞, exporting `gamma_star`, or reproducing a calibrated controller.
- Do not use this for: Post-hoc visualization or statistical standardization that leaves the synthesized problem and saved `gamma_star` unchanged.
- Notes: Cross-model reports must state that raw-coordinate `gamma_star` values can retain model-scale dependence. Any reporting-only normalization must be saved separately and must never rewrite controller inputs or gains.

## Controlled Decoding KV Cache

- Decision: Default transformer KV caching to off during evaluated-model generation for Original and every steering method. Retain explicit `--kv-cache on` support for appendix comparisons; the two modes use separate evaluation and result directories.
- Why: Under the current activation-hook feedback implementation, cache-on and cache-off decoding produce materially different controller trajectories and benchmark outcomes; cache state is therefore part of the intervention semantics rather than only a runtime optimization.
- Use this when: Generating behavior, capability, ID, or distribution-shift evaluations with the shared steering pipeline.
- Do not use this for: Uncontrolled evaluator and judge models, which may use KV caching because no activation feedback policy is attached.
- Notes: Record the evaluated-model cache state in every stage log. Scorer-model cache behavior is separate and does not define the evaluated-model condition.
