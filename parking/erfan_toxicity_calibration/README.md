# controller_calibration

## Method

- Use 50 toxic and 50 non-toxic RealToxicityPrompts records to fit an eight-dimensional, layer-specific PCA coordinate system for each model.
- Whiten every retained coordinate with fit-split variance, then fit adjacent-layer linear dynamics by ridge regression.
- Use 50 disjoint RTP calibration records to measure held-out dynamics residuals and retain PCA disturbance channels explaining 95% of residual variance.
- Synthesize A-LQR and the final H∞ controller once per model; cache the coordinates, gains, disturbance geometry, `gamma_star`, and prompt identifiers for all downstream Erfan benchmarks.
- Plot `gamma_star` and the median retained disturbance rank for each model.

## Variables

- Data/input: pinned RealToxicityPrompts train split.
- Sessions/groups: DistilGPT-2 and Qwen-2.5 0.5B, 1.5B, 7B, and 14B.
- Labels/targets: non-toxic minus toxic reduced-state direction.
- Signals/features/measures: whitened reduced activations, fitted dynamics, residual covariance factors, `gamma_star`, and disturbance rank.
- Parameters/thresholds: seed 2151; rank 8; ridge 0.001; 95% disturbance variance; 2% H∞ deployment margin.
- Outputs: `plots/controller_calibration.pdf` and `.png`.

## Statistics

- Tests/models: layer-wise PCA whitening, ridge-regression dynamics, and finite-horizon H∞ feasibility bisection.
- Null hypothesis: not applicable; this is calibration and descriptive reporting.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: disturbance components are retained until cumulative explained variance reaches 95%; `gamma_star` is the smallest feasible attenuation value within `1e-5`.
- What the statistic means: lower `gamma_star` gives larger calibration-only robust steerability `1/gamma_star`.
- Why this statistic is appropriate here: H∞ requires an empirically calibrated disturbance geometry that is isolated from all benchmark test prompts.

## Legends

- X axis: model.
- Y axis: `gamma_star` in the left panel and median layer-wise disturbance rank in the right panel.
- Color/value: dark green denotes H∞ attenuation; midnight blue denotes retained disturbance dimension.
- Grouping: one bar per model.
- Ordering/sorting: increasing Qwen parameter count after DistilGPT-2.
- Lines/markers/labels: bars only.
- Panels: attenuation boundary and disturbance dimension.

## Interpretation

- Every model produced a feasible final H∞ controller. `gamma_star` ranges from
  0.705 for Qwen-2.5-1.5B to 1.219 for DistilGPT-2, corresponding to
  `S_rob = 1/gamma_star` from 0.820 to 1.418. Median retained disturbance rank
  ranges from 5.5 to 7 of the eight reduced coordinates.

## Notes

- These are archived results under their recorded original calibration. This unit now exposes plotting only; the superseded 50-prompt runner is retained at `ref/paper_benchmark_50/`. Historical caches are not upgraded or used as current benchmark results.

- Fit, calibration, and benchmark records are disjoint by prompt identifier.
- Controller caches are intentionally ignored by Git and reused by every Erfan benchmark.

## References

- `robust_steerability/control/h_infinity.py`
- `robust_steerability/calibration/disturbances.py`
- `DECISIONS.md`
