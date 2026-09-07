# model_scale_residuals

## Method

- Run GPT-2 and Qwen checkpoints on 50 within-distribution prompts and 50 prompts from each selected handcrafted OOD family.
- Measure each layer’s residual-block update norm divided by its output-state norm, then average within early, middle, and late depth thirds.
- Plot trends, model-by-depth matrices, scaling summaries, and the top-three OOD grid.

## Variables

- Data/input: fixed sentiment-style within-distribution and OOD prompt banks.
- Sessions/groups: GPT-2 family and Qwen-2.5 checkpoints across parameter scale.
- Labels/targets: adversarial, long-context, and Spanish shifts.
- Signals/features/measures: relative residual-block update by normalized depth regime.
- Parameters/thresholds: 50 expanded prompts per condition; maximum length 128 for the recommended run.
- Outputs: `plots/network_size_residual_*` files.

## Statistics

- Tests/models: descriptive means by model, condition, and depth regime.
- Null hypothesis: no inferential test is performed.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: OOD amplification is the OOD mean minus the within-distribution mean.
- What the statistic means: positive amplification indicates larger residual updates under the named shift.
- Why this statistic is appropriate here: normalized depth and relative residual permit descriptive comparisons across architectures.

## Legends

- X axis: parameter count or model order, depending on the output.
- Y axis: mean relative residual or OOD-minus-ID residual.
- Color/value: early, middle, and late depth regimes use distinct fixed colors.
- Grouping: model family and OOD condition.
- Ordering/sorting: increasing parameter count within the stored model catalog.
- Lines/markers/labels: connected model trends, bars, or heatmap cells depending on output.
- Panels: parallel depth regimes or OOD conditions.

## Interpretation

- The outputs ask whether OOD mismatch grows with model scale and where in normalized depth it is strongest.

## Notes

- This controller-independent unit was renamed and preserved without recomputation.

## References

- `DECISIONS.md`

# benchmark_ood_sweep

## Method

- Sample 50 records from RealToxicityPrompts, Jigsaw, Civil Comments, ToxicChat, and matched/unmatched MMLU subjects.
- Measure last-token residual-block updates for every selected model and summarize early, middle, late, and overall depth.
- Compare every OOD subset with its dataset-matched ID baseline.

## Variables

- Data/input: benchmark-backed prompt pools named above.
- Sessions/groups: models, benchmark subsets, and normalized depth regimes.
- Labels/targets: ID RTP, ID MMLU subject, and OOD subsets.
- Signals/features/measures: relative residual update and OOD-minus-ID amplification.
- Parameters/thresholds: seed 2151; 50 prompts; eight MMLU OOD subjects by default.
- Outputs: `plots/benchmark_ood_sweep_grid.png` and `plots/benchmark_ood_sweep_amplification.png`.

## Statistics

- Tests/models: descriptive means and matched-baseline differences.
- Null hypothesis: no inferential test is performed.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: positive difference denotes OOD amplification.
- What the statistic means: each value averages prompt-level residuals within a model/subset/depth cell.
- Why this statistic is appropriate here: it keeps dataset shift and concept shift comparisons tied to explicit ID baselines.

## Legends

- X axis: model.
- Y axis: mean relative residual or OOD-minus-ID difference.
- Color/value: benchmark subset.
- Grouping: early, middle, and late panels.
- Ordering/sorting: stored model-catalog order.
- Lines/markers/labels: lines for absolute residuals and bars for amplification.
- Panels: three depth regimes.

## Interpretation

- The sweep identifies the strongest benchmark-backed shift for each model.

## Notes

- None yet.

## References

- `robust_steerability/benchmarks/ood.py`

# paper_exports

## Method

- Convert cached model-scale and benchmark-sweep summaries into condensed CSV/LaTeX tables and paper-sized plots.
- Rank each model’s strongest overall and late-layer OOD condition and export the corresponding differences.

## Variables

- Data/input: cached summary JSON and CSV from the two analyses above.
- Sessions/groups: models, OOD subsets, and depth regimes.
- Labels/targets: strongest OOD amplification.
- Signals/features/measures: absolute residual, OOD-minus-ID differences, and parameter count.
- Parameters/thresholds: maximum difference determines the selected condition.
- Outputs: condensed cached tables and `plots/*amplification*` / `plots/*strongest_ood_deltas*` figures.

## Statistics

- Tests/models: descriptive maxima and means; no inferential test.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: select the largest recorded OOD-minus-ID difference.
- What the statistic means: the export reports the empirically strongest tested condition, not an unbiased population maximum.
- Why this statistic is appropriate here: it creates a compact summary of a fixed exploratory sweep.

## Legends

- X axis: parameter count or model.
- Y axis: residual amplification.
- Color/value: depth regime or selected OOD subset.
- Grouping: model family.
- Ordering/sorting: increasing model size.
- Lines/markers/labels: paper-export specific.
- Panels: defined by the generated output.

## Interpretation

- These files summarize, rather than extend, the model-scale residual measurements.

## Notes

- None yet.

## References

- `erfan_model_scale_residuals.py`
- `erfan_benchmark_ood_sweep.py`
