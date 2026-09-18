# benchmark_summary

## Method

- Join the fresh truthfulness and in-distribution toxicity tables by model and steering method.
- Plot truthful-times-informative percentage and RTP continuation toxicity side by side using the shared four-method encoding.

## Variables

- Data/input: `parking/erfan_truthfulness_benchmark/plots/results.csv` and `parking/erfan_id_toxicity_benchmark/plots/results.csv`.
- Sessions/groups: five model sizes × four methods.
- Labels/targets: truthfulness and toxicity control.
- Signals/features/measures: TruthfulQA truthful-times-informative percentage and mean RTP toxicity percentage.
- Parameters/thresholds: both source experiments use 50 held-out prompts per benchmark.
- Outputs: `plots/benchmark_summary.pdf` and `.png`.

## Statistics

- Tests/models: no new statistical model; the figure displays source-unit means.
- Null hypothesis: no inferential test is performed.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: higher is better for truthfulness and lower is better for toxicity.
- What the statistic means: the two panels summarize behavioral utility and steering effectiveness.
- Why this statistic is appropriate here: both panels use the same model and method ordering and therefore provide a compact cross-benchmark view.

## Legends

- X axis: model.
- Y axis: truthful-times-informative percentage or RTP toxicity percentage.
- Color/value: gray Original, midnight blue A-LQR, dark orange S-PID, dark green H-infinity.
- Grouping: four bars per model.
- Ordering/sorting: increasing Qwen scale after DistilGPT-2.
- Lines/markers/labels: bars only.
- Panels: truthfulness and ID toxicity.

## Interpretation

- The fresh 50-prompt runs show strong ID toxicity suppression from A-LQR and
  H∞ while retaining or improving truthful-times-informative score for all Qwen
  scales; DistilGPT-2 is the exception, where H∞ lowers that truthfulness metric.

## Notes

- This is a derived summary; the source units own prompt-level caches and detailed figures.

## References

- `parking/erfan_truthfulness_benchmark/README.md`
- `parking/erfan_id_toxicity_benchmark/README.md`
