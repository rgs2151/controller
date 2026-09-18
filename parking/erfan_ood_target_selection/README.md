# ood_condition_ranking

## Method

- Load the ten-condition A-LQR remaining-target-error summary from `parking/ood_explore`.
- Sort conditions by median remaining error and display the interquartile range.
- Preserve Erfan’s selected per-model top-one, top-three, and global-three target tables alongside the plot.

## Variables

- Data/input: `parking/ood_explore/plots/lqr_ood_failure_summary.json`.
- Sessions/groups: ID plus nine OOD prompt families.
- Labels/targets: remaining semantic target error after A-LQR.
- Signals/features/measures: median, first quartile, and third quartile across 50 prompts.
- Parameters/thresholds: strongest condition is the largest median remaining error.
- Outputs: `plots/ood_condition_ranking.pdf`, `.png`, and four selected-target CSV files.

## Statistics

- Tests/models: descriptive medians and interquartile ranges.
- Null hypothesis: no inferential test is performed.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: conditions are ranked from largest to smallest median.
- What the statistic means: larger remaining error means weaker A-LQR correction.
- Why this statistic is appropriate here: the median and IQR are robust to the heavy-tailed adversarial prompt responses.

## Legends

- X axis: condition sorted by median remaining error.
- Y axis: remaining A-LQR target error as a percentage of unsteered error.
- Color/value: color families distinguish ID, structured shifts, language/context shifts, and adversarial corruption.
- Grouping: one bar per condition.
- Ordering/sorting: descending median error.
- Lines/markers/labels: whiskers show the IQR; stars mark highlighted candidates.
- Panels: one panel.

## Interpretation

- The ranking chooses OOD conditions for the multimodel steering benchmark without looking at its generated-text outcomes.

## Notes

- The selected CSVs are preserved Erfan artifacts and are not recomputed by the new H∞ runs.

## References

- `parking/ood_explore/README.md`
- `parking/erfan_model_scale_residuals/README.md`

# ood_family_ranking

## Method

- Run Qwen-2.5-0.5B-Instruct on ten handcrafted prompts per OOD family.
- Compute a full last-token decoder-block Jacobian and relative one-step residual at every layer.
- Compare each family against the fixed within-distribution prompt set in early, middle, and late layer thirds with violin distributions and Welch tests.

## Variables

- Data/input: ten fixed within-distribution prompts and ten fixed prompts for each OOD family.
- Sessions/groups: within-distribution versus one OOD family at a time.
- Labels/targets: early, middle, and late normalized depth regimes.
- Signals/features/measures: relative Jacobian linearization residual.
- Parameters/thresholds: Qwen-2.5-0.5B-Instruct; maximum length 32; VJP chunk 64; alpha 0.05.
- Outputs: `plots/ood_family_ranking_violin_grid.png`.

## Statistics

- Tests/models: two-sided Welch independent-samples t-test for every family-regime cell.
- Null hypothesis: mean relative residual is equal for within-distribution and OOD prompts.
- Alternative hypothesis: the two means differ.
- Thresholds/decision rule: a star denotes `p < 0.05`; otherwise the panel is labeled `n.s.`.
- What the statistic means: each test asks whether that prompt family changes average relative residual in the selected layer regime.
- Why this statistic is appropriate here: the two fixed prompt groups are independent and may have unequal variances, although the small handcrafted sample limits generalization.

## Legends

- X axis: within-distribution and OOD.
- Y axis: mean relative residual within the named layer regime.
- Color/value: teal is within-distribution and orange is OOD.
- Grouping: rows are OOD families; columns are layer regimes.
- Ordering/sorting: fixed family order.
- Lines/markers/labels: points are prompts, white circles are means, and stars mark `p < 0.05`.
- Panels: ten families × three depth regimes.

## Interpretation

- The grid separates prompt families whose mismatch is localized to different transformer depths.

## Notes

- This independent residual analysis does not use H∞ and therefore was preserved rather than rerun.

## References

- `robust_steerability/modeling/jacobians.py`
