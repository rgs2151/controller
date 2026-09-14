# h_infinity_calibration_heatmap

## Method

- Load the pinned TruthfulQA generation validation split and take positions 101–200 from its seed-42 permutation as 100 calibration questions.
- Reuse Gemma-2-2B's frozen 200-false/200-true semantic target, 35-Jacobian averaged dynamics, and rank-8 coordinates. Refit the H∞ disturbance channels so `D[k]D[k]ᵀ` exactly reconstructs the centered empirical covariance of the separate calibration residuals.
- Fix the setpoint multiplier at the published Gemma A-LQR value λ=3 and fix R=1.
- Synthesize the 32 H∞ controllers formed by crossing Q/R over `0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6` and Qf/R over `0.01, 0.0316, 0.1, 0.316`.
- Generate the same 100 questions under five independent seeds for every controller and score every completion with the pinned TruthfulQA True and Helpful judges.
- Compute T×I within each repetition, select the largest five-repetition mean, and break exact ties by mean Informative, mean True, smaller Q/R, then smaller Qf/R.
- Write the complete grid to CSV, outline the selected cell in the heatmap, and freeze that selected controller in the benchmark-artifact unit.

## Variables

- Data/input: TruthfulQA `generation` validation split, revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490`; 100 unique calibration questions.
- Sessions/groups: 32 H∞ controllers, each evaluated in five generation repetitions of the same 100 questions.
- Labels/targets: positive-minus-negative TruthfulQA MC2 semantic target; exact binary True and Helpful judge decisions.
- Signals/features/measures: generated completions, repetition-level True, Informative, T×I, and synthesized gamma-star.
- Parameters/thresholds: Gemma-2-2B revision `c5ebcd40d208330abc697524c919956e692655cf`; λ=3; R=1; generation seed 42 with a 100,000 repetition stride; only exact case-insensitive `yes` scores 1.
- Outputs: `plots/h_infinity_calibration_heatmap.{pdf,png}`, `plots/h_infinity_gamma_star_heatmap.{pdf,png}`, and `plots/h_infinity_calibration_grid.csv`.

## Statistics

- Tests/models: descriptive mean and standard error of repetition-level binary-judge percentages over five independent generations; no inferential test.
- Null hypothesis: not tested.
- Alternative hypothesis: not tested.
- Thresholds/decision rule: maximize mean repetition-level T×I with the deterministic tie-break stated above.
- What the statistic means: each cell is the mean across five products of that repetition's True and Informative rates.
- Why this statistic is appropriate here: the controller is calibrated against the same combined behavioral score used in the TruthfulQA benchmark while averaging generation randomness.

## Legends

- X axis: terminal-state cost ratio Qf/R.
- Y axis: running-state cost ratio Q/R.
- Color/value: the calibration heatmap encodes mean T×I from 0% to 100%; the gamma-star heatmap encodes the minimum feasible gamma for the same controller cell. Both print the underlying value in every cell.
- Grouping: each square is one H∞ cost configuration evaluated on 100 questions × five seeds.
- Ordering/sorting: Q/R increases from 0.01 to 31.6; Qf/R increases from 0.01 to 0.316.
- Lines/markers/labels: the dark-red outline marks the selected cell.
- Panels: two separate 8-by-4 heatmaps with identical cost-grid organization.

## Interpretation

- Under the corrected covariance-factor disturbance fit, the grid selects Q/R=0.1 and Qf/R=0.1.
- The selected cell reaches 62.34 ± 1.32 T×I on the five 100-question calibration repetitions, with 80.80 ± 1.24 True and 77.20 ± 1.80 Informative.
- Gamma-star ranges from 8.07 to 131.08 across the grid; the behavior-selected cell has gamma-star 23.53.
- Its publication-scale result is reported only by `benchmarks/truthfulness/`; calibration performance is not treated as benchmark performance.

## Notes

- The setpoint multiplier is fixed at 3 and is not swept.
- The terminal-state cost ratio is restricted below 1; the running-state cost ratio retains the full range through 31.6.
- Repeating the same 100 questions measures generation variation; it does not create 500 unique calibration questions.
- All 16,000 generations and both 16,000-row judge caches remain inside this unit.
- Old H∞ controller and generation identities are rejected rather than adapted to the corrected calibration protocol.

## References

- `benchmarks/truthfulness/cache/gemma2b/calibrations/h_infinity/selected/`
- `benchmarks/truthfulness/`
- `robust_steerability/control/h_infinity.py`

# truthfulness_table

## Method

- Optionally compare Original, S-PID, A-LQR, adopted H∞, Kaz H∞, and the calibration-selected H∞ controller on the first 100 questions of the same seed-42 TruthfulQA permutation.
- Score completions with the pinned True and Helpful judges and report True, Informative, and their aggregate-rate product T×I.
- Keep this comparison as a parked diagnostic; the publication-scale five-by-817 H∞ result is owned only by `benchmarks/truthfulness/`.

## Variables

- Data/input: 100 unique TruthfulQA generation questions disjoint from the 100 calibration questions.
- Sessions/groups: Original, S-PID, A-LQR, adopted H∞, Kaz H∞, and selected H∞.
- Labels/targets: pinned binary True and Helpful judge decisions.
- Signals/features/measures: completions, True, Informative, and T×I percentages.
- Parameters/thresholds: adopted H∞ uses λ=3, Q=0.1, R=1, Qf=1; Kaz H∞ uses λ=3, Q=0.5, R=1, Qf=0.3.
- Outputs: `plots/truthfulness_table.md`, `plots/truthfulness_table.csv`, and `plots/summary.json` when explicitly generated.

## Statistics

- Tests/models: descriptive percentages over one matched 100-question set; no inferential test.
- Null hypothesis: not tested.
- Alternative hypothesis: not tested.
- Thresholds/decision rule: every included method must have exactly 100 generations and paired judge outputs.
- What the statistic means: True and Informative are marginal positive-judge rates; T×I is their aggregate product.
- Why this statistic is appropriate here: it provides a cheap matched diagnostic without being confused with the publication-scale benchmark.

## Legends

- X axis: none; this output is a table.
- Y axis: none.
- Color/value: cells report percentages; no color is used.
- Grouping: rows identify controller configurations.
- Ordering/sorting: Original, S-PID, A-LQR, adopted H∞, Kaz H∞, then selected H∞.
- Lines/markers/labels: no interval is reported for the one-set diagnostic.
- Panels: none.

## Interpretation

- The adopted and Kaz configurations are exploratory parked comparisons only.
- No new 100-question diagnostic table was generated after the current calibration; the current iteration uses the grid-selected controller directly in the publication-scale benchmark.
- The grid-selected controller's publication-scale evaluation is reported only by `benchmarks/truthfulness/`, not by this diagnostic table.

## Notes

- This diagnostic must not populate the main benchmark table.
- The completed five-by-817 H∞ benchmark remains independent of the adopted and Kaz diagnostic configurations.

## References

- `benchmarks/truthfulness/`
- `robust_steerability/source_methods/`
