# qwen_layer_residuals

## Method

- Load the first eight SST-2 training sentences and Qwen-2.5-0.5B-Instruct.
- At every decoder block, compute the full last-token Jacobian and compare `J[k] x[k]` with the observed next-layer hidden state.
- Divide the residual norm by the observed next-state norm and average across prompts by layer.
- Write the layer summary to the ignored cache and plot the mean curve.

## Variables

- Data/input: GLUE SST-2 train split.
- Sessions/groups: eight prompts and all Qwen decoder layers.
- Labels/targets: SST-2 labels are retained as metadata but not used in the plotted average.
- Signals/features/measures: full Jacobian, observed next-layer state, residual norm, and relative residual.
- Parameters/thresholds: maximum length 32; VJP chunk 64; seed 0.
- Outputs: `plots/qwen_layer_residuals.png`.

## Statistics

- Tests/models: direct relative residual with layer-wise mean, median, standard deviation, minimum, and maximum.
- Null hypothesis: no inferential test is performed.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: larger relative residual indicates poorer local linear prediction.
- What the statistic means: the plotted value is average mismatch divided by observed state magnitude.
- Why this statistic is appropriate here: normalization makes decoder layers with different state norms comparable within this model.

## Legends

- X axis: decoder layer index.
- Y axis: mean relative residual across prompts.
- Color/value: one residual curve.
- Grouping: one point per layer.
- Ordering/sorting: input-to-output layer order.
- Lines/markers/labels: connected circular markers.
- Panels: one panel.

## Interpretation

- Peaks identify layers where the local zero-reference Jacobian approximation is least accurate.

## Notes

- The preserved notebook and additional PNGs are Erfan’s exploratory views of the same residual question.
- This unit is independent of H∞ and was reorganized without changing its numeric artifacts.

## References

- `robust_steerability/modeling/jacobians.py`
- `parking/residual_checks/README.md`
