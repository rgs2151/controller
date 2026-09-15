# bench_table

## Method

- Load completed controlled-decoding summaries for RTP, Jigsaw, TruthfulQA ID, and Spanish TruthfulQA.
- Align summaries by the fixed model and method order.
- Render one Markdown table and one TeX table from the same row objects; absent scorer outputs remain `TBD`.

## Variables

- Data/input: `benchmarks/*/results/kv_cache_off/` summary JSON files.
- Sessions/groups: model-method pairs with five benchmark repetitions when complete.
- Labels/targets: RTP and Jigsaw toxicity, Dist-2, and PPL; TruthfulQA True, Informative, instruction relevance, and fluency.
- Signals/features/measures: repetition means and standard errors; True and Informative are percentages, while instruction relevance and fluency are independent 0–2 AXBench judgments.
- Parameters/thresholds: evaluated-model KV cache disabled; Spanish receives the ID-calibrated controller.
- Outputs: `plots/bench_table.md` and `plots/bench_table.tex`.

## Statistics

- Tests/models: descriptive means and standard errors across five repetitions.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: missing summaries or unrun scorers render as `TBD`.
- What the statistic means: each populated cell estimates one model-method metric under one prompt distribution.
- Why this statistic is appropriate here: every metric is summarized over the benchmark's repeated evaluation protocol without combining independent scorers.

## Legends

- X axis: table columns name the metric and preferred direction.
- Y axis: model-method rows.
- Color/value: none.
- Grouping: model, then method.
- Ordering/sorting: fixed manuscript order.
- Lines/markers/labels: `TBD` means unrun.
- Panels: toxicity table first, truthfulness table second.

## Interpretation

- The table is populated one independently scored benchmark result at a time.
- TruthfulQA reports True, Informative, instruction relevance, and fluency as four separate measurements.

## Notes

- Regenerate both synchronized outputs with `python figs/bench_table/bench_table.py` after summaries change.

## References

- `benchmarks/truthfulness/`.
- `benchmarks/toxicity/`.
- `DECISIONS.md`.
