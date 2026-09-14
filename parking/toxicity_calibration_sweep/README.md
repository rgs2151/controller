# h_infinity_toxicity_heatmap

## Method

- Read the completed Gemma-2-2B H∞ calibration grid for RealToxicityPrompts.
- Arrange mean classifier toxicity by the tested `Q/R` rows and `Qf/R` columns.
- Mark the configuration selected under the joint toxicity and perplexity rule.

## Variables

- Data/input: `benchmarks/toxicity/cache/gemma2b/calibrations/h_infinity/selected/selection.json`.
- Model/task: Gemma-2-2B toxicity mitigation.
- Measure: percentage of generated continuations classified as toxic.
- Parameters: five 100-prompt calibration repetitions; evaluated-model KV cache off.
- Output: `plots/h_infinity_toxicity_heatmap.{pdf,png}`.

## Statistics

- None; the heatmap descriptively reports the mean percentage across five repetitions.
- Selection rule: lowest mean toxicity among configurations whose mean perplexity is at most 1.5 times Original.

## Legends

- X axis: terminal-state cost ratio `Qf/R`.
- Y axis: running-state cost ratio `Q/R`.
- Color/value: mean classifier toxicity percentage; lower is better.
- Outline: selected configuration.

## Interpretation

- Shows where toxicity mitigation is strongest within the fixed H∞ calibration grid.

## Notes

- This plot reads calibration results and does not rerun generation or controller synthesis.

## References

- `benchmarks/toxicity/README.md`.

# h_infinity_perplexity_heatmap

## Method

- Read the same completed grid and arrange mean prompt-inclusive perplexity by `Q/R` and `Qf/R`.
- Mark the selected configuration.

## Variables

- Data/input: the selected Gemma-2-2B toxicity calibration record.
- Measure: mean prompt-inclusive perplexity from the pinned evaluator.
- Output: `plots/h_infinity_perplexity_heatmap.{pdf,png}`.

## Statistics

- None; the heatmap descriptively reports the mean across five 100-prompt repetitions.
- Decision threshold: eligible configurations must remain at or below 1.5 times Original perplexity.

## Legends

- X axis: terminal-state cost ratio `Qf/R`.
- Y axis: running-state cost ratio `Q/R`.
- Color/value: mean perplexity; lower is better.
- Outline: selected configuration.

## Interpretation

- Shows the fluency cost used to constrain toxicity-based controller selection.

## Notes

- Perplexity is a calibration guard, not the selection objective by itself.

## References

- `benchmarks/toxicity/README.md`.

# h_infinity_gamma_star_heatmap

## Method

- Load each synthesized H∞ controller from the same grid.
- Arrange its minimum feasible attenuation boundary `gamma_star` by `Q/R` and `Qf/R`.
- Mark the behaviorally selected configuration.

## Variables

- Data/input: `benchmarks/toxicity/cache/gemma2b/calibrations/h_infinity/selected/grid/controllers/`.
- Measure: saved minimum feasible `gamma_star` from unchanged H∞ synthesis.
- Output: `plots/h_infinity_gamma_star_heatmap.{pdf,png}`.

## Statistics

- None; this is a direct visualization of synthesized controller values.

## Legends

- X axis: terminal-state cost ratio `Qf/R`.
- Y axis: running-state cost ratio `Q/R`.
- Color/value: minimum feasible `gamma_star`.
- Outline: configuration selected by toxicity subject to the perplexity guard.

## Interpretation

- Shows how the H∞ feasibility boundary changes over the behavioral calibration grid.

## Notes

- `gamma_star` is not used as a standalone behavioral selection score.

## References

- `benchmarks/toxicity/README.md`.
