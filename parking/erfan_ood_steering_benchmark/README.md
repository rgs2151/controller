# ood_toxicity_across_models

## Method

- Select 50 prompts each from the longest Jigsaw comments, longest ToxicChat conversations, and the MMLU other-concepts mixture identified by Erfan’s residual screen.
- Generate matched-seed 40-token continuations with Original, A-LQR, S-PID, and final H∞, then score toxic-class probability.
- Plot method means by model separately for each shift condition.

## Variables

- Data/input: Jigsaw, ToxicChat, MMLU, and the pinned toxicity classifier.
- Sessions/groups: five models × three OOD conditions × four methods.
- Labels/targets: cross-dataset, conversational/length, and concept shifts.
- Signals/features/measures: mean continuation toxic-class probability.
- Parameters/thresholds: 50 prompts per condition; seed 2151; 40 generated tokens.
- Outputs: `plots/results.csv`, `plots/ood_toxicity_across_models.pdf`, and `.png`.

## Statistics

- Tests/models: descriptive arithmetic means.
- Null hypothesis: no inferential test is performed.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: lower continuation toxicity is better.
- What the statistic means: each bar averages the same 50 prompt-level toxicity probabilities.
- Why this statistic is appropriate here: matched prompts isolate controller differences within every model-condition pair.

## Legends

- X axis: model.
- Y axis: continuation toxicity percentage.
- Color/value: gray Original, midnight blue A-LQR, dark orange S-PID, dark green H-infinity.
- Grouping: four neighboring bars per model.
- Ordering/sorting: increasing Qwen scale after DistilGPT-2.
- Lines/markers/labels: bars only.
- Panels: Jigsaw long, ToxicChat long, and MMLU concept shift.

## Interpretation

- Transfer is strongly model-condition dependent. H∞ reduces Jigsaw toxicity
  by 64.5% at 0.5B and 48.8% at 1.5B, but only 0.6% at 14B. On ToxicChat it
  reduces toxicity by about 99% at 0.5B, 1.5B, and 14B while increasing it for
  DistilGPT-2 and 7B; this is the clearest failure exposed by the rerun.

## Notes

- These are archived results under their recorded original calibration. This unit now exposes plotting only; fresh 50-prompt execution is `python parking/paper_benchmark_50/paper_benchmark_50.py --devices cuda:0,cuda:1`. Historical caches are not upgraded or used as current benchmark results.

- MMLU prompts are evaluated for collateral toxic generation, not MMLU answer accuracy, in this unit.

## References

- Target selection: `parking/erfan_ood_target_selection/`.
- Historical implementation: `ref/erfan_applied_controller/`.

# ood_toxicity_model_metrics

## Method

- Reshape the same OOD result table into a model-by-condition grid and plot absolute continuation toxicity for all methods.

## Variables

- Data/input: `plots/results.csv` from this unit.
- Sessions/groups: model rows, OOD-condition columns, method bars.
- Labels/targets: absolute OOD toxic-class probability.
- Signals/features/measures: mean toxicity percentage.
- Parameters/thresholds: the fixed 50-prompt benchmark sample.
- Outputs: `plots/ood_toxicity_model_metrics.pdf` and `.png`.

## Statistics

- Tests/models: descriptive arithmetic means; no additional model or test.
- Null hypothesis: no inferential test is performed.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: lower is better.
- What the statistic means: each bar is one model-condition-method mean.
- Why this statistic is appropriate here: the grid exposes model-specific failures hidden by cross-model averaging.

## Legends

- X axis: method.
- Y axis: continuation toxicity percentage.
- Color/value: the shared four-method color mapping.
- Grouping: rows are models; columns are OOD conditions.
- Ordering/sorting: fixed model and condition order.
- Lines/markers/labels: bars only.
- Panels: fifteen model-condition combinations.

## Interpretation

- The absolute grid shows that the largest controller separation occurs on
  Jigsaw for the smaller Qwen models and on ToxicChat for DistilGPT-2 and 7B.
  All MMLU concept-shift toxicity means are below 0.01%.

## Notes

- None yet.

## References

- `plots/results.csv`.

# ood_reduction_model_metrics

## Method

- Compute `100 × (Original − method) / Original` from the same matched result rows and plot the reduction grid.

## Variables

- Data/input: `plots/results.csv` from this unit.
- Sessions/groups: model rows, OOD-condition columns, method bars.
- Labels/targets: relative reduction from each model-condition Original mean.
- Signals/features/measures: percentage toxicity reduction.
- Parameters/thresholds: Original is exactly zero reduction by definition.
- Outputs: `plots/ood_reduction_model_metrics.pdf` and `.png`.

## Statistics

- Tests/models: descriptive paired-condition percentage changes.
- Null hypothesis: no inferential test is performed.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: positive values improve on Original; negative values worsen it.
- What the statistic means: the normalized change controls for different baseline toxicity levels across conditions.
- Why this statistic is appropriate here: absolute classifier rates differ substantially between natural toxicity and concept-shift prompts.

## Legends

- X axis: method.
- Y axis: reduction from Original in percent.
- Color/value: the shared four-method color mapping.
- Grouping: rows are models; columns are OOD conditions.
- Ordering/sorting: fixed model and condition order.
- Lines/markers/labels: bars only.
- Panels: fifteen model-condition combinations.

## Interpretation

- Positive H∞ bars identify transfer beyond the unsteered baseline; direct comparison with A-LQR tests robust-control advantage.

## Notes

- Very small Original toxicity values can make relative changes unstable; inspect the absolute grid alongside this figure.

## References

- `plots/ood_toxicity_model_metrics.pdf`.
