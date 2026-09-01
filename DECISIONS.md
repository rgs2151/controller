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

- Decision: Compare models only after whitening reduced state coordinates with calibration statistics, standardizing control and semantic-output coordinates, expressing layer position as normalized depth, and weighting per-layer costs by the normalized depth increment.
- Why: Raw activation scales, hidden dimensions, and layer counts differ across models and otherwise confound disturbance-gain comparisons.
- Use this when: Comparing robust steerability, control energy, residual propagation, or controller performance across checkpoints or model families.
- Do not use this for: Within-model diagnostic plots that explicitly report raw activation coordinates.
- Notes: Always report whether a result uses normalized or raw coordinates.
