# refresh_data

## Method

- Read the synchronized Markdown tables in `figs/bench_table/` for TruthfulQA, Spanish TruthfulQA, MGSM, L-CiteEval, and HarmBench.
- Parse each reported mean and standard error, retain the means required by the imported plotting interfaces, and map manuscript method labels to the plot scripts' internal labels.
- Exclude ODESteer from TruthfulQA because the imported figures explicitly omit that method.
- Write four long-form CSV inputs to this unit's `cache/` directory.

## Variables

- Data/input: current benchmark tables under `figs/bench_table/`.
- Labels/targets: model, method, distribution, language, context length, and jailbreak template.
- Signals/features/measures: TruthfulQA True, instruction relevance, and fluency; MGSM accuracy and three steering measures; L-CiteEval answer recall, citation F1, and overall steering; HarmBench ASR and three steering measures.
- Outputs: `cache/truthfulqa_figure_data.csv`, `cache/mgsm_transfer_results.csv`, `cache/lciteeval_context_results.csv`, and `cache/harmbench_refusal_results.csv`.

## Statistics

- None; this script transfers already-computed descriptive means from synchronized result tables.
- No hypothesis test, fitted model, threshold, or additional aggregation is introduced.

## Legends

- None; this script produces tabular inputs rather than a figure.

## Interpretation

- The cached CSVs are synchronized plotting views of the current benchmark tables, not independent result sources.

## Notes

- The attached archive's original CSVs remain recoverable from `ref/robust_LQR_source.zip`.

## References

- `figs/bench_table/truthfulness/`, `figs/bench_table/mgsm/`, `figs/bench_table/lciteeval/`, and `figs/bench_table/harmful/`.

# plot_truth_quality_frontier

## Method

- Plot English TruthfulQA and Spanish-transfer TruthfulQA in separate figures.
- Represent each model by color and each steering method by marker shape.
- Compute response quality as `min(instruction relevance / 2, fluency / 2)` and quality-adjusted truthfulness (QAT) as `True (%) × quality`.
- Overlay constant-QAT contours and emphasize H∞ with an outer gold ring.

## Variables

- Data/input: `cache/truthfulqa_figure_data.csv`.
- Groups: ID and OOD split, model, and method.
- X axis: normalized response quality in `[0, 1]`.
- Y axis: TruthfulQA True percentage.
- Outputs: `plots/pdf/truthfulqa-{id,ood}-quality-frontier.pdf` and matching PNG files.

## Statistics

- None; the output is descriptive.
- QAT contours are deterministic products of the reported means and are not inferential confidence regions.

## Legends

- Color: foundation model.
- Marker: steering method.
- Gold outline: H∞.
- Dashed contours: equal QAT values.
- Panels: one figure each for ID and Spanish OOD evaluation.

## Interpretation

- Points farther up and right jointly retain response quality and improve truthfulness.
- Equal-QAT contours expose whether a truthfulness gain compensates for reduced relevance or fluency.

## Notes

- ODESteer is omitted by the imported figure design.

## References

- `figs/bench_table/truthfulness/truthfulqa.md` and `truthfulqa_spanish.md`.

# plot_truthfulqa_model_summary

## Method

- Compute each method's QAT and subtract the same model's unsteered Original QAT.
- Place one model on each row and show all available steering methods as markers.
- Emphasize H∞ and visually compare it with the strongest non-H∞ method for each model.

## Variables

- Data/input: `cache/truthfulqa_figure_data.csv`.
- X axis: change in QAT from Original, in QAT points.
- Y axis: model identity.
- Grouping: method markers within each model row.
- Outputs: `plots/pdf/truthfulqa-{id,ood}-model-summary.pdf` and matching PNG files.

## Statistics

- None; differences are computed directly from reported means.
- No paired uncertainty is shown because the imported analysis lacks paired per-question QAT estimates.

## Legends

- Marker shape and color: steering method.
- Vertical zero line: no change from the same model's Original response.
- Connector: H∞ relative to the strongest reported alternative.
- Panels: separate ID and OOD figures.

## Interpretation

- Positive values indicate higher quality-adjusted truthfulness than the unsteered model.

## Notes

- ODESteer is excluded consistently with the imported code.

## References

- `plot_truth_quality_frontier.py` for the QAT definition.

# plot_truthfulqa_gain_matrix

## Method

- For every model-method cell, subtract Original True percentage and Original QAT.
- Display the two changes as aligned matrices and summarize H∞'s QAT margin over the strongest available non-H∞ method.
- Leave unavailable model-method cells blank rather than imputing values.

## Variables

- Data/input: `cache/truthfulqa_figure_data.csv`.
- Rows: models.
- Columns: steering methods.
- Color/value: change in True percentage points or QAT points from Original.
- Outputs: `plots/pdf/truthfulqa-{id,ood}-gain-matrix.pdf` and matching PNG files.

## Statistics

- None; all cells are descriptive differences of reported means.
- The diverging scale is centered at zero, the decision boundary for improvement versus degradation.

## Legends

- Blue: negative change from Original.
- White: zero change.
- Red: positive change from Original.
- Right panel: H∞ QAT margin over the best non-H∞ method.
- Panels: truthfulness gain, quality-adjusted gain, and H∞ margin.

## Interpretation

- The matrices separate raw truthfulness gains from gains that survive a relevance-and-fluency quality adjustment.

## Notes

- Missing methods remain blank; ODESteer is excluded.

## References

- `plot_truth_quality_frontier.py` for the QAT definition.

# plot_transfer_results

## Method

- For MGSM, macro-average accuracy and steering measures across the five transfer languages within each model-method pair.
- Define transfer quality as the minimum of Spanish relevance, instruction relevance, and fluency after division by two, then plot it against accuracy.
- Also display per-language accuracy by method and model.
- For L-CiteEval, plot answer recall, citation F1, and overall steering against 8K, 16K, and 32K context length for every method.

## Variables

- Data/input: `cache/mgsm_transfer_results.csv` and `cache/lciteeval_context_results.csv`.
- MGSM groups: model, method, and Chinese/French/Japanese/Swahili/Telugu.
- L-CiteEval groups: model, method, and context length.
- Outputs: `plots/pdf/mgsm-transfer-overview.pdf`, `plots/pdf/lciteeval-context-robustness.pdf`, `plots/pdf/lciteeval-context-robustness-drop-32k.pdf`, and matching PNG files.

## Statistics

- None; the figures show descriptive means.
- MGSM macro-averages give equal weight to each language.
- Constant-QAA contours are deterministic products of MGSM accuracy and transfer quality.

## Legends

- Color: model.
- Marker and line style: method.
- Gold outline: H∞.
- MGSM axes: transfer quality versus accuracy, plus per-language accuracy.
- L-CiteEval axes: context length versus answer recall, citation F1, or overall steering.

## Interpretation

- MGSM shows the capability-steering tradeoff across language shift.
- L-CiteEval shows whether answer and citation quality persist as context length grows.

## Notes

- The plotting layout and visual encodings are preserved from the supplied archive; the 32K label now reflects its completed status.
- `lciteeval-context-robustness-drop-32k` is an otherwise identical reporting variant restricted to the 8K and 16K results; the complete 8K/16K/32K figure remains unchanged.

## References

- `figs/bench_table/mgsm/` and `figs/bench_table/lciteeval/`.

# plot_harmbench_results

## Method

- Plot Direct and five frozen human-jailbreak template ASRs for Original, A-LQR, and H∞.
- Order templates by baseline difficulty and shade the ASR margin between H∞ and the stronger baseline.
- Average each method across the six displayed conditions and plot mean ASR against mean safe-concept relevance.
- Render one visually identical figure per model so results from different model sizes are not mixed.
- Also render a compact combined figure as one model per row with shared method encodings, common axes, and only the mean-ASR reduction annotation retained.

## Variables

- Data/input: `cache/harmbench_refusal_results.csv`.
- Templates: Direct, John persona, DNE nonresponse, Jailbreak Bot, YOJA/Nona roleplay, and APM programmer.
- Left axes: template and attack success rate percentage.
- Right axes: mean ASR and mean safe-concept relevance.
- Outputs: one PDF and PNG per HarmBench model plus `harmbench-robust-refusal-combined.pdf` and its matching PNG under `plots/`.

## Statistics

- None; the figure is descriptive.
- Template and frontier means give equal weight to each displayed condition.
- The annotated ASR reduction is relative to the lower mean ASR of Original and A-LQR.

## Legends

- Line and marker: method.
- Gold outline: H∞.
- Gold shading: robustness margin to the stronger baseline.
- Left panel: template-level ASR profile.
- Right panel: mean safety Pareto comparison.

## Interpretation

- Lower ASR and higher safe-concept relevance indicate safer, better-aligned behavior.

## Notes

- The archive contained a one-model input. The current unit preserves the same visual format and writes one figure for each of the three available models.

## References

- `figs/bench_table/harmful/harmbench_full.md`.

# render_all

## Method

- Refresh all four CSV inputs, clear only this unit's prior plot outputs, and invoke the five imported plotting scripts in a fixed order.

## Variables

- Inputs: synchronized benchmark tables and this unit's plotting scripts.
- Outputs: all PDFs under `plots/pdf/` and all PNGs under `plots/figures/`.

## Statistics

- None; this is an orchestration script and adds no calculations.

## Legends

- See the producing plot script sections above.

## Interpretation

- A successful run means every benchmark figure was rebuilt from the same current table snapshot.

## Notes

- The original source archive, original figures, manuscript fragments, and assets are retained under `ref/` and `assets/`.

## References

- All scripts documented above.
