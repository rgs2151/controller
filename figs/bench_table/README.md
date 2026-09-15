# bench_table_kv_cache_on

## Method

- Read only completed summaries under `benchmarks/*/results/kv_cache_on/`.
- Align values by the fixed model and method order and render identical Markdown and TeX cells.
- Preserve the earlier full-run results as a historical cache-on condition; missing cells remain `TBD`.

## Variables

- Data/input: cache-on RTP, Jigsaw, TruthfulQA-ID, and Spanish summary JSON files.
- Sessions/groups: model-method pairs; five benchmark repetitions where complete.
- Labels/targets: RTP and Jigsaw toxicity, Dist-2, and PPL; TruthfulQA T×I, True, Info, AXBench instruction relevance, AXBench fluency, and MMLU.
- Signals/features/measures: summary means and standard errors; AXBench instruction relevance and fluency are separate 0--2 `gpt-4o-mini` judgments made in strict structured batches of 20.
- Parameters/thresholds: evaluated-model KV cache on; evaluator cache state is separate.
- Outputs: `plots/bench_table_kv_cache_on.md` and `plots/bench_table_kv_cache_on.tex`.

## Statistics

- Tests/models: descriptive means and standard errors supplied by the evaluation unit.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: absent summaries or uncomputed expanded Jigsaw metrics render as `TBD`.
- What the statistic means: each cell is one cache-on model-method benchmark estimate.
- Why this statistic is appropriate here: the renderer does not combine incompatible cache conditions.

## Legends

- X axis: table columns name metrics and preferred direction.
- Y axis: model-method rows.
- Color/value: none.
- Grouping: model, then method.
- Ordering/sorting: fixed manuscript order.
- Lines/markers/labels: `TBD` means unrun.
- Panels: toxicity first, truthfulness second.

## Interpretation

- These are preserved historical cache-on results and are not the active controlled-decoding results.

## Notes

- The manuscript currently points explicitly to this historical TeX file until the cache-off table is populated and reviewed.

## References

- `benchmarks/truthfulness/` and `benchmarks/toxicity/`.

# bench_table_kv_cache_off

## Method

- Read only completed summaries under `benchmarks/*/results/kv_cache_off/`.
- Render RTP ID and Jigsaw OOD toxicity, Dist-2, and PPL beside the truthfulness benchmark without importing cache-on values.
- Keep all unexecuted cells as `TBD`; the prepared toxicity run targets only Original, S-PID, A-LQR, and H∞ on Gemma-2-2B.

## Variables

- Data/input: cache-off RTP, Jigsaw, TruthfulQA-ID, and Spanish summary JSON files.
- Sessions/groups: five × 1,000 RTP prompts, five × 1,000 Jigsaw prompts, and five × 817 TruthfulQA prompts when run.
- Labels/targets: classifier toxicity on generated continuations in both toxicity distributions; Dist-2 and PPL output-quality measurements; truthfulness metrics.
- Signals/features/measures: mean ± SE toxicity, Dist-2, and PPL for both RTP and Jigsaw, plus T×I, True, Info, AXBench instruction relevance (0--2), AXBench fluency (0--2), and truthfulness-table MMLU accuracy.
- Parameters/thresholds: evaluated-model KV cache off; Jigsaw receives the RTP-selected controller with no recalibration.
- Outputs: `plots/bench_table_kv_cache_off.md` and `plots/bench_table_kv_cache_off.tex`.

## Statistics

- Tests/models: descriptive repetition means and standard errors; truthfulness-table MMLU uses prompt-level Bernoulli standard error.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: toxic probability greater than 0.5 is toxic; incomplete or absent result files render as `TBD`.
- What the statistic means: RTP is ID toxicity control and output quality; Jigsaw is cross-dataset toxicity transfer and output quality under the same frozen controller.
- Why this statistic is appropriate here: toxicity, Dist-2, and PPL are computed identically across the two toxicity prompt populations.

## Legends

- X axis: table columns name metrics and preferred direction.
- Y axis: model-method rows.
- Color/value: none.
- Grouping: model, then method.
- Ordering/sorting: fixed manuscript order; the requested toxicity methods are Original, S-PID, A-LQR, and H∞.
- Lines/markers/labels: `TBD` means unrun.
- Panels: toxicity first, truthfulness second.

## Interpretation

- This is the active controlled-decoding table. It contains the completed
  Gemma-2-2B RTP and Jigsaw results for Original, S-PID, A-LQR, and H∞; other
  model-method cells remain `TBD`.

## Notes

- Regenerate both condition files with `python figs/bench_table/bench_table.py` after summaries change.
- Markdown and TeX are always written from the same row objects.

## References

- `robust_steerability/benchmarks/toxicity.py`.
- `DECISIONS.md`.
