# truthfulness_model_metrics

## Method

- Sample the same 50 TruthfulQA generation questions and 50 MMLU five-shot questions for every model and controller.
- Generate with Original, A-LQR, S-PID, and final H∞ using matched per-prompt seeds; MMLU uses deterministic two-token decoding.
- Score TruthfulQA completions with the official AllenAI truth and informativeness judge models and score MMLU by parsed answer letter.
- Report percentages and binomial standard errors, then plot four metrics for every model.

## Variables

- Data/input: TruthfulQA validation and MMLU test/dev splits.
- Sessions/groups: five model sizes × four steering methods.
- Labels/targets: truthful, informative, truthful-times-informative, and correct MMLU response.
- Signals/features/measures: judge yes/no outputs and exact MMLU correctness.
- Parameters/thresholds: 50 prompts per benchmark; five MMLU shots; seed 2151; 50 generated TruthfulQA tokens.
- Outputs: `plots/results.csv`, `plots/truthfulness_model_metrics.pdf`, and `.png`.

## Statistics

- Tests/models: Bernoulli means with analytic standard errors; first-order uncertainty propagation for the product metric.
- Null hypothesis: no inferential test is performed; each plotted proportion descriptively estimates method performance on the fixed sample.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: judge output beginning with “yes” is positive; MMLU is correct only when the parsed letter matches the gold index.
- What the statistic means: bar height is the percentage of successful records; error bars are one standard error.
- Why this statistic is appropriate here: every metric is binary at the prompt level and methods are evaluated on identical records.

## Legends

- X axis: Original, A-LQR, S-PID, and H-infinity.
- Y axis: percentage for the metric named above each column.
- Color/value: gray is Original, midnight blue A-LQR, dark orange S-PID, and dark green H-infinity.
- Grouping: rows are models and columns are metrics.
- Ordering/sorting: increasing Qwen scale after DistilGPT-2.
- Lines/markers/labels: bars with one-standard-error whiskers.
- Panels: truthful-times-informative, truthful, informative, and MMLU.

## Interpretation

- H∞ changes truthful-times-informative score by -5.6, +0.1, +5.8, +20.2,
  and +14.3 percentage points relative to Original from DistilGPT-2 through
  Qwen-2.5-14B. MMLU differences remain within six points inside each model,
  so this 50-question run does not show a large utility collapse.

## Notes

- The run uses one paper-aligned protocol for every model; it does not mix Erfan’s earlier lexical smoke metric with official judge scores.

## References

- Historical implementation: `ref/erfan_applied_controller/`.
- Controller calibration: `parking/erfan_toxicity_calibration/`.
