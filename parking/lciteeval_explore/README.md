# lciteeval_calibration_heatmaps

## Method

- Load the completed Qwen2.5-3B-Instruct L-CiteEval H∞ calibration record.
- Align the 12 candidates into four `Q/R` rows and three `Qf/R` columns with `R = 1` and `λ = 1.5` fixed.
- Display the mean concept relevance, instruction relevance, fluency, and per-response AXBench harmonic-mean score saved for each candidate.
- Mark the configuration selected by the maximum mean per-response AXBench harmonic mean.
- Write synchronized PNG and PDF heatmaps.

## Variables

- Data/input: `cache/selection.json`, copied from the completed Remote 1 benchmark artifact `~/robust-steering-cache/lciteeval/qwen25_3b_instruct/calibrations/h_infinity/selected/selection.json`.
- Model/task: Qwen2.5-3B-Instruct on the L-CiteEval calibration task.
- Calibration prompts: 50 held-out AlpacaEval prompts, one deterministic generation per candidate.
- Rows: `Q/R ∈ {0.01, 0.1, 1, 10}`.
- Columns: `Qf/R ∈ {0.01, 0.1, 0.316227766}`.
- Fixed parameters: `R = 1` and `λ = 1.5`.
- Selected controller: `Q/R = 0.01`, `Qf/R = 0.01`, and `γ★ = 2.3054`.
- Outputs: `plots/lciteeval_calibration_heatmaps.png` and `plots/lciteeval_calibration_heatmaps.pdf`.

## Statistics

- None; the heatmaps descriptively report the saved mean scores for each calibration candidate.
- Selection rule: maximize the mean of the per-response harmonic mean of AXBench concept relevance, instruction relevance, and fluency.
- The selected statistic measures joint steering quality and becomes zero for a response when any of its three component scores is zero.
- This statistic is appropriate because calibration requires the target concept to appear without sacrificing task relevance or language quality.

## Legends

- X axis: terminal-cost ratio `Qf/R`.
- Y axis: running-state-cost ratio `Q/R`.
- Color/value: the named AXBench score on its native 0–2 scale; all four panels share the same color normalization.
- Grouping: one cell per H∞ cost configuration.
- Ordering/sorting: both cost axes increase numerically from the upper-left corner.
- Lines/markers/labels: a red outline and star mark the selected configuration; cell text gives the exact saved mean.
- Panels: concept relevance, instruction relevance, fluency, and their per-response harmonic-mean overall steering score.

## Interpretation

- The chosen low-running-cost, low-terminal-cost controller has the highest overall steering score (`0.444`) despite not maximizing every component separately.
- The grid shows that the selected controller combines strong instruction relevance and fluency with sufficient target-concept expression to outperform the other joint configurations.

## Notes

- This unit reads completed calibration output and never reruns generation or controller synthesis.
- The selected `γ★` belongs to the chosen controller; it is not the heatmap selection objective.
- The same selected controller is transferred unchanged to the 8K, 16K, and 32K L-CiteEval conditions.

## References

- `benchmarks/lciteeval/README.md`.
- `figs/bench_table/lciteeval/`.
