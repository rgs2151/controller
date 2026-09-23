# Qwen2.5-3B-Instruct context-length transfer

## Method

- Read the finalized L-CiteEval full-results table.
- Plot the Qwen2.5-3B-Instruct results at matched 8K and long-context 16K conditions.
- Render answer recall and citation F1 as separate panels with one trajectory per method.
- Write a vector PDF and matching PNG to `plots/`.

## Variables

- Data/input: `figs/bench_table/lciteeval/lciteeval_full.md`.
- Sessions/groups: context length and steering method for Qwen2.5-3B-Instruct.
- Labels/targets: Original, S-PID, A-LQR, and H∞.
- Signals/features/measures: answer recall (%) and citation F1 (%).
- Parameters/thresholds: 8K is the matched condition; 16K is the long-context condition.
- Outputs: `plots/figure_long_context_qwen2_5_3b.pdf` and `plots/figure_long_context_qwen2_5_3b.png`.

## Statistics

- Tests/models: descriptive visualization of the finalized full-sample means.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: higher answer recall and higher citation F1 are better.
- What the statistic means: answer recall measures recovered gold-answer content; citation F1 balances citation precision and recall.
- Why this statistic is appropriate here: the two measures expose answer retention and citation quality under context-length shift without conflating them.

## Legends

- X axis: 8K matched and 16K long-context conditions.
- Y axis: answer recall (%) or citation F1 (%), depending on panel.
- Color/value: method identity; H∞ uses project teal `#398197`.
- Grouping: one line per method.
- Ordering/sorting: 8K precedes 16K.
- Lines/markers/labels: each method has a stable color, line style, and marker; the official Qwen logo identifies the model.
- Panels: answer recall and citation F1.

## Interpretation

- The figure shows which controllers retain answer content and citation quality when the same questions move from 8K to 16K context.
- The key comparison is H∞ against Original, S-PID, and A-LQR at each context length.

## Notes

- Error bars are intentionally omitted because the requested figure compares the finalized means; uncertainty remains documented in the source table.
- Typography follows the repository-wide Arial standard.

## References

- `figs/bench_table/lciteeval/lciteeval_full.md`.
- `figs/logos/qwen_transparent.png`.

# Llama-3.2-1B-Instruct context-length transfer

## Method

- Read the finalized L-CiteEval full-results table.
- Plot the Llama-3.2-1B-Instruct results at matched 8K and long-context 16K conditions.
- Render answer recall and citation F1 as separate panels with one trajectory per method.
- Write a vector PDF and matching PNG to `plots/`.

## Variables

- Data/input: `figs/bench_table/lciteeval/lciteeval_full.md`.
- Sessions/groups: context length and steering method for Llama-3.2-1B-Instruct.
- Labels/targets: Original, A-LQR, and H∞.
- Signals/features/measures: answer recall (%) and citation F1 (%).
- Parameters/thresholds: 8K is the matched condition; 16K is the long-context condition.
- Outputs: `plots/figure_long_context_llama3_2_1b.pdf` and `plots/figure_long_context_llama3_2_1b.png`.

## Statistics

- Tests/models: descriptive visualization of the finalized full-sample means.
- Null hypothesis: not applicable.
- Alternative hypothesis: not applicable.
- Thresholds/decision rule: higher answer recall and higher citation F1 are better.
- What the statistic means: answer recall measures recovered gold-answer content; citation F1 balances citation precision and recall.
- Why this statistic is appropriate here: the two measures expose answer retention and citation quality under context-length shift without conflating them.

## Legends

- X axis: 8K matched and 16K long-context conditions.
- Y axis: answer recall (%) or citation F1 (%), depending on panel.
- Color/value: method identity; H∞ uses project teal `#398197`.
- Grouping: one line per method.
- Ordering/sorting: 8K precedes 16K.
- Lines/markers/labels: each method has a stable color, line style, and marker; the official Meta/Llama logo identifies the model.
- Panels: answer recall and citation F1.

## Interpretation

- The figure shows which controllers retain answer content and citation quality when the same questions move from 8K to 16K context.
- The key comparison is H∞ against Original and A-LQR at each context length.

## Notes

- S-PID is absent because the finalized Llama-3.2-1B table contains no S-PID row.
- Error bars are intentionally omitted because the requested figure compares the finalized means; uncertainty remains documented in the source table.
- Typography follows the repository-wide Arial standard.

## References

- `figs/bench_table/lciteeval/lciteeval_full.md`.
- `figs/logos/llama_transparent.png`.
