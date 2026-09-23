# figure_0_overall

## Method

- Read only the finalized Markdown tables under `figs/bench_table/` for TruthfulQA, Spanish TruthfulQA, HarmBench, MGSM, and L-CiteEval.
- Define the four shifts as English-to-Spanish TruthfulQA transfer, direct-to-human-jailbreak HarmBench transfer, multilingual MGSM evaluation, and 8K-to-16K L-CiteEval transfer.
- For each model and condition, select the highest-performing reported non-H∞ row as the best competitor. For MGSM language shift, `Original` is not a steering competitor and is explicitly excluded, leaving the strongest reported alternative steering method. HarmBench attack success rate is converted to safe-response rate as `100 - ASR`; all other measures are already higher-is-better.
- Compute each bar as `sum(model score × model parameters in billions) / sum(model parameters in billions)` across the models reported for that benchmark.
- Overlay each model-family logo at that model's unaggregated score. Logo display size increases monotonically with parameter count; the separate model-family legend uses equal logo sizes.
- Write a vector PDF, a 300-DPI PNG, and an audit CSV containing every selected method, individual score, parameter count, and weighted mean.

## Variables

- Data/input: `figs/bench_table/truthfulness/truthfulqa.md`, `truthfulqa_spanish.md`, `harmful/harmbench_summary.md`, `mgsm/mgsm_full.md`, and `lciteeval/lciteeval_summary.md`.
- Conditions: Truthfulness groups are English and Spanish translation; adversarial groups are direct requests and the mean across five adversarial templates; language groups are Chinese, French, Japanese, Swahili, and Telugu; context groups are 8K and 16K.
- Measures: truthful-response percentage; safe-response percentage; exact-answer accuracy percentage; citation F1 percentage.
- Model-size weights: GPT-2 XL 1.5B, Llama-3-8B 8B, Qwen-2.5-14B 14B, OLMo-2-32B 32B; HarmBench Llama models 1B/3B/8B; MGSM Qwen/Phi/Granite 4B/3.8B/2B; L-CiteEval Qwen/Llama 3B/1B.
- Colors: H∞ `#398197`; best competitor `#A7ADB2`.
- Outputs: `plots/figure_0_overall.pdf`, `plots/figure_0_overall.png`, and `plots/figure_0_overall_values.csv`.

## Statistics

- Tests/models: None; this is a descriptive aggregation of finalized benchmark point estimates.
- Null hypothesis: None.
- Alternative hypothesis: None.
- Thresholds/decision rule: For each model-condition pair, the best competitor is the maximum higher-is-better score among every reported method except H∞. HarmBench is inverted before this comparison.
- What the statistic means: Bar height is the model-parameter-weighted mean performance within one benchmark condition.
- Why this statistic is appropriate here: It implements the requested size-weighted cross-model summary while retaining every individual model result as a logo mark.

## Legends

- X axis: English and Spanish translation; Direct and Adversaries; Chinese, French, Japanese, Swahili, and Telugu; 8K and 16K.
- Y axis: Truthful responses, safe responses, accuracy, or citation F1, all in percent and all higher-is-better. Language-shift accuracy is capped at 80%; context-shift citation F1 is capped at 10%. Every panel includes the same proportional 5% unlabeled margin below its zero line so zero-valued model marks remain unobstructed.
- Color/value: gray bars are the best competitor; teal bars are H∞.
- Grouping: each x-axis label names the concrete evaluation condition rather than using ID/OOD shorthand.
- Ordering/sorting: shifts follow truthfulness, adversarial, language, and context; languages follow Chinese, French, Japanese, Swahili, and Telugu.
- Lines/markers/labels: unboxed logos mark individual model scores; logo size increases with model parameters; the model-family legend uses a common display size.
- Panels: the first pair contains truthfulness and adversarial shifts; additional horizontal space separates it from the language and context pair. The language panel is widened so all five labels remain on one line while retaining the common rendered bar width.
- Legend placement: the bar-series legend is centered above the full figure; the compact model-family legend remains at right.

## Interpretation

- Compare the teal and gray bar heights within each condition while using logo positions to see whether the aggregate is broad across models or driven by a particular scale.

## Notes

- Only rows present in the finalized benchmark tables are eligible; exploratory or superseded runs are excluded.
- “Best competitor” may be a different method for different models and conditions; the audit CSV records each selection. In language shift, the unsteered `Original` row is never eligible.

## References

- Shared logo provenance: `figs/logos/SOURCES.md`.
- Final benchmark tables: `figs/bench_table/`.
