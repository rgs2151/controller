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
- Colors: H∞ uses the exact project teal `#007C7C`; best competitor uses the
  exact aggregate gray `#737B80`.
- Outputs: `plots/figure_0_overall.pdf`, `plots/figure_0_overall.png`, and `plots/figure_0_overall_values.csv`.
- Alternative display: `figure_0_overall_boxplot.py` writes
  `plots/figure_0_overall_boxplot.pdf` and `.png` using the identical data,
  logos, colors, and axes. It replaces each aggregate bar with an unfilled box
  spanning the model-score quartiles, a median line, and min/max whiskers;
  model logos retain their exact individual scores. Paired boxes are separated
  with compact within-box logo jitter so that series membership stays legible.
  Logos on H∞ boxes use a 20%-darker version of `#007C7C`, while logos on
  best-competitor boxes use a 20%-darker version of `#737B80`; native company
  colors remain in the model-family legend. Recoloring uses the source logo's
  alpha and luminance mask rather than alpha alone, preserving light and
  multicolor silhouettes such as the Microsoft four-pane mark. Box borders,
  medians, whiskers, and caps use the exact opaque series colors.
  In this alternative display, the adversarial panel is cropped to 70--100%
  to resolve the tightly clustered safe-response results. Arial typography,
  trimmed outward-offset spines, and larger labels follow the shared figure
  style. Nested headings organize panels as Parallel Steering versus
  Orthogonal Steering, then by shift, distribution (ID/OOD), and concrete
  evaluation condition. OOD headings use the project red `#8B1E1E`.

## Statistics

- Tests/models: None; this is a descriptive aggregation of finalized benchmark point estimates.
- Null hypothesis: None.
- Alternative hypothesis: None.
- Thresholds/decision rule: For each model-condition pair, the best competitor is the maximum higher-is-better score among every reported method except H∞. HarmBench is inverted before this comparison.
- What the statistic means: Bar height is the model-parameter-weighted mean performance within one benchmark condition.
- Why this statistic is appropriate here: It implements the requested size-weighted cross-model summary while retaining every individual model result as a logo mark.

## Legends

- X axis: English and Spanish; Direct and Adversaries; Chinese, French, Japanese, Swahili, and Telugu; 8K and 16K.
- Y axis: Truthful responses, safe responses, accuracy, or citation F1, all in percent and all higher-is-better. Language-shift accuracy is capped at 60%; context-shift citation F1 is capped at 10%. Every panel includes the same proportional 5% unlabeled margin below its zero line so zero-valued model marks remain unobstructed.
- Color/value: gray bars are the best competitor; teal bars are H∞.
- Grouping: each x-axis label names the concrete evaluation condition rather than using ID/OOD shorthand.
- Ordering/sorting: shifts follow truthfulness, adversarial, language, and context; languages follow Chinese, French, Japanese, Swahili, and Telugu.
- Lines/markers/labels: unboxed logos mark individual model scores; logo size increases with model parameters; the model-family legend uses a common display size.
- Panels: the first pair contains truthfulness and adversarial shifts under `Parallel Steering`; the language and context pair appears under `Orthogonal Steering`. Each shift has a nested ID/OOD heading, with language shift labeled OOD once across all five languages. Explicit equal-width spacer columns separate all four benchmark axes while preserving the compact 16.2-inch canvas and the established physical width of the language panel.
- Legend placement: the box-series legend is centered below the full figure; the compact native-color model-family legend remains at right.

## Interpretation

- Compare the teal and gray bar heights within each condition while using logo positions to see whether the aggregate is broad across models or driven by a particular scale.

## Notes

- Only rows present in the finalized benchmark tables are eligible; exploratory or superseded runs are excluded.
- “Best competitor” may be a different method for different models and conditions; the audit CSV records each selection. In language shift, the unsteered `Original` row is never eligible.

## References

- Shared logo provenance: `figs/logos/SOURCES.md`.
- Final benchmark tables: `figs/bench_table/`.
