# bench_table

## Method

- Read completed toxicity and truthfulness summary JSON files from the active benchmark evaluation unit.
- Align results by the fixed model and method ordering used in the manuscript.
- Insert each available mean and standard error into its named metric column; leave unrun cells as `TBD`.
- Render the same row objects to Markdown and TeX so the two formats cannot contain different values.
- Write one Markdown document and one TeX fragment containing the toxicity and truthfulness tables.

## Variables

- Data/input: `parking/bench_evaluations/cache/results/<behavior>/<model>/<method>.json`.
- Sessions/groups: Gemma-2-2B, Llama-3-8B, and Qwen-2.5-14B; Original, ITI, ActAdd, Mean-AcT, Linear-AcT, PID-AcT, ODESteer, S-PID, A-LQR, and H∞.
- Labels/targets: toxicity, Dist-2, MMLU, PPL, truthful-times-informative, True, Info, Spanish, adversarial, and long-context performance.
- Signals/features/measures: metric means and standard errors from completed benchmark summaries.
- Parameters/thresholds: five 1,000-prompt RTP repetitions, five complete 817-question TruthfulQA repetitions, and one shared 1,000-question five-shot MMLU set.
- Outputs: `plots/bench_table.md` and `plots/bench_table.tex`.

## Statistics

- Tests/models: descriptive means and standard errors supplied by the benchmark summaries; no inferential test is performed here.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: a missing result or optional metric is displayed as `TBD`; a completed result missing a required metric is an error.
- What the statistic means: each cell reports the estimated benchmark metric and its recorded sampling uncertainty.
- Why this statistic is appropriate here: the unit preserves the evaluation unit's summaries without recomputing or combining incompatible observations.

## Legends

- X axis: table columns name the benchmark metrics and preferred direction.
- Y axis: table rows are model-method pairs.
- Color/value: no color encoding; cells contain mean ± SE or `TBD`.
- Grouping: methods are grouped within model blocks.
- Ordering/sorting: fixed model order, then Original through H∞.
- Lines/markers/labels: arrows indicate whether larger or smaller values are preferred.
- Panels: the TeX and Markdown documents contain toxicity first and truthfulness second.

## Interpretation

- Compare steering methods only within the same model and metric.
- `TBD` denotes missing computation, not zero performance.

## Notes

- Run `python figs/bench_table/bench_table.py` after new benchmark summaries are completed.
- Edit neither output by hand; both formats are regenerated together from the same values.
- Historical 50-prompt tables are retained under `ref/paper_benchmark_50/` and are not inputs.

## References

- `parking/bench_evaluations/`.
- `ref/2604.19018v1.pdf`.
- `/home/dev/controller/paper/iclr2026_conference.tex`.
