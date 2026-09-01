# residual_geometry

## Method

- Load `data/representation_dynamics/residual_trajectories.npz` for one exact model checkpoint and steering behavior.
- Align each one-step dynamics residual with its hidden-state deviation, prompt split, shift condition, and normalized transformer depth.
- Measure normalized residual magnitude for every record and layer as `||xi_k||_2 / (||x_k||_2 + 1e-12)`.
- Use only calibration-split residuals to fit a separate principal-component disturbance basis at each layer. Center calibration residuals and retain the smallest number of right-singular-vector directions explaining at least 95% of their variance.
- Project centered test-split residuals into the corresponding calibration basis and measure the fraction of each residual's squared energy explained by that basis.
- Average normalized residual magnitude and explained residual energy across test records within each shift condition. Report the retained calibration rank as a fraction of the state dimension.
- Write a three-panel vector PDF containing residual magnitude, held-out disturbance-subspace coverage, and retained rank across normalized depth.

## Variables

- Data/input: `data/representation_dynamics/residual_trajectories.npz` following the exact schema in `data/README.md`.
- Sessions/groups: test prompt trajectories grouped as `id`, `paraphrase`, `ood`, or `adversarial`; calibration prompts are used only to fit the disturbance bases.
- Labels/targets: exact `model_id` and `behavior` stored in the input; shift-condition labels identify plotted test groups.
- Signals/features/measures: one-step residual `xi_k = x[k+1] - A[k] x[k] - B[k] u[k]`, hidden-state deviation `x_k`, normalized residual magnitude, calibration residual principal components, held-out explained residual energy, and retained-rank fraction.
- Parameters/thresholds: variance-retention threshold `0.95`; numerical denominator floor `1e-12`; normalized depth in `[0, 1]`.
- Outputs: local cache `cache/residual_geometry.pkl` and tracked plot `plots/residual_geometry.pdf`.

## Statistics

- Tests/models: layer-wise principal component analysis fitted by singular-value decomposition on calibration residuals; all plotted group summaries are arithmetic means across held-out test records.
- Null hypothesis: none; this unit performs descriptive uncertainty characterization and does not conduct a significance test.
- Alternative hypothesis: none.
- Thresholds/decision rule: retain the minimum number of calibration components whose cumulative explained variance is at least 95%.
- What the statistic means: normalized residual magnitude measures one-step local-model error relative to deviation-state magnitude; held-out explained energy measures how much test residual energy lies in the disturbance subspace learned from calibration prompts; retained-rank fraction measures the dimensionality needed to represent 95% of calibration residual variance relative to the analyzed state dimension.
- Why this statistic is appropriate here: the robust controller requires a disturbance channel that captures both the magnitude and directional structure of representation-dynamics error, so the unit separately measures error size, subspace transfer, and dimensionality before controller evaluation.

## Legends

- X axis: transformer layer position normalized to `[0, 1]` in all panels.
- Y axis: panel A shows mean normalized residual magnitude; panel B shows mean held-out explained residual energy; panel C shows retained calibration rank divided by state dimension.
- Color/value: midnight blue is in-distribution test prompts, dark green is paraphrase shift, dark red is out-of-distribution shift, purple is adversarial shift, and black is the calibration retained-rank fraction.
- Grouping: colored lines aggregate held-out test records within shift condition; the black line is computed from calibration records.
- Ordering/sorting: layers are displayed in ascending normalized depth; shift conditions use the fixed order ID, paraphrase, OOD, adversarial when present.
- Lines/markers/labels: solid lines show condition means without markers; panel A contains the shared condition legend.
- Panels: A, normalized residual magnitude; B, held-out residual energy captured by calibration directions; C, calibration disturbance-subspace rank fraction.

## Interpretation

- Panel A describes whether representation-dynamics mismatch grows or changes location across prompt shifts.
- Panel B describes whether calibration residual directions continue to cover unseen residuals even when residual magnitude changes.
- Panel C describes whether the calibrated disturbance geometry is low-dimensional relative to the analyzed state.
- The unit does not by itself establish controller performance or a cross-model robust-steerability relationship.

## Notes

- The analysis-ready input has not been generated yet, so no plot or numerical result is claimed.
- Run with `--recompute` only when the residual input or analysis definition changes; otherwise the unit reuses its local cache.

## References

- `data/README.md`
- `DECISIONS.md`
- `STYLE.md`
- The manuscript section “What Does Representation-Dynamics Uncertainty Look Like?”
