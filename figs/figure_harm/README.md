# figure_harm

## Method

- Load the frozen HarmBench template-level results for Llama 3.2 1B, Llama 3.2 3B, and Llama 3.1 8B.
- Summarize each model–method pair across the direct request and five jailbreak templates.
- Plot mean versus worst-case attack rejection, template-level attack success profiles, and the aggregate safety frontier in one 1×3 figure.
- Write a vector PDF and matching PNG to `plots/`.

## Variables

- Data/input: `figs/benchmark_figures/cache/harmbench_refusal_results.csv`.
- Sessions/groups: model, steering method, and jailbreak template.
- Labels/targets: Original, A-LQR, and H-infinity.
- Signals/features/measures: attack success rate, attack rejection, and safe-concept relevance.
- Parameters/thresholds: rejection is `100 - ASR`; worst-case rejection uses the maximum ASR across the six conditions.
- Outputs: `plots/figure_harm.pdf` and `plots/figure_harm.png`.
  The preserved geometric-marker version remains unchanged; the iterative
  logo-based variant is written separately as `figure_harm_logos.pdf/.png`.
  The active main figure is `figure_harm_main.pdf/.png` (template profile plus
  safety frontier), while the detached rejection panel is retained as
  `figure_harm_rejection.pdf/.png`.

## Statistics

- Tests/models: descriptive aggregation only.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: lower ASR and higher rejection/relevance are better.
- What the statistic means: mean rejection captures average robustness, while worst-case rejection captures the weakest template condition.
- Why this statistic is appropriate here: the panels jointly expose average robustness, template sensitivity, and the safety–relevance tradeoff.

## Legends

- X axis: panel-specific mean rejection, jailbreak template, or mean ASR.
- Y axis: panel-specific worst-case rejection, template ASR, or mean safe-concept relevance.
- Color/value: model family scale in the middle/right panels and steering method in the left panel.
- Grouping: results are grouped by model and method.
- Ordering/sorting: jailbreak templates use the benchmark's fixed direct-plus-five-template order.
- Lines/markers/labels: color and line style encode method; circle, triangle, and square encode the 1B, 3B, and 8B Llama models; marker area increases with model scale.
- Logo variant: the official shared Llama mark replaces geometric shapes and
  is tinted by method. Log-scaled logo size distinguishes 1B, 3B, and 8B;
  H∞ uses exact project teal `#007C7C`. The top native-color model legend uses
  the same ordered size hierarchy, while the bottom method legend uses dots.
  Original uses gray with reduced opacity; A-LQR uses a distinct muted violet
  (`#6558A6`) so it remains separable without competing with the teal focal
  series.
- Panels: rejection summary, robustness profile, and safety frontier.

## Interpretation

- The figure compares whether H-infinity improves robust refusal across model scales and jailbreak conditions while retaining safe-concept relevance.
- The key comparison is H-infinity against the strongest available non-H-infinity result.

## Notes

- This is the initial consolidated layout; panel-specific visual revisions are intentionally deferred to subsequent iterations.
- `figure_harm_logos` is the active iterative variant. It uses enlarged
  MGSM-style axis, tick, template, and legend typography without overwriting
  the preserved initial figure.
- The final organization separates the former first panel from the main
  composition. In `figure_harm_main`, a blank marginal-height spacer aligns
  the jailbreak-template axis with the frontier axis, extra negative-y room
  prevents zero-ASR Llama marks from clipping, and the x-axis label is offset
  below the two-line template labels. The compact near-equal panel footprints
  leave room for neighboring manuscript panels; explicit endpoint padding and
  nonuniform categorical spacing keep Direct, the long central labels, and
  Jailbreak Bot clear of the axis boundaries without stretching the panel.

## References

- `figs/benchmark_figures/scripts/plot_harmbench_results.py`.
- `figs/bench_table/harmful/`.
