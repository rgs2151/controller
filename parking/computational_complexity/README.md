# computational_complexity

## Method

- Use the frozen Llama-3.2-1B-Instruct HarmBench refusal artifacts to compare Original generation, the deployed full-state A-LQR policy, and the deployed reduced-state H-infinity policy without modifying any benchmark result.
- Profile offline synthesis on one NVIDIA GeForce RTX 5090. Clear Python and CUDA allocator caches, synchronize CUDA before and after each measurement, and alternate methods across 20 trials. The deployed A-LQR recursion uses the saved 16-layer, 2,048-dimensional dynamics; H-infinity uses the saved 16-layer, 8-dimensional robust-control problem. A separate matched-core A-LQR timing uses the same 8-dimensional problem as H-infinity.
- Reproduce the production H-infinity feasibility bisection exactly and record the feasible interval after every iteration. Refit disturbance geometry for 100 seeded bootstrap resamples of the 50 calibration residual records, then synthesize H-infinity and retain gamma-star, convergence, iteration count, and upper-bound expansions.
- Measure deployment on the same 16 frozen HarmBench direct prompts. Force exactly 50 newly generated tokens per prompt, batch eight prompts, disable the evaluated-model KV cache, rotate method order across seven synchronized trials, and divide elapsed generation time by the exact 800 tokens produced per method and trial.
- Reset peak CUDA statistics before each deployment trial. Report absolute peak allocated VRAM and allocation above the resident-model baseline.
- Write raw trials and provenance to `cache/`, the four-panel figure to `plots/computational_complexity.{pdf,png}`, and synchronized summaries to `plots/synthesis_summary.csv` and `plots/inference_summary.csv`.

## Variables

- Data/input: Llama-3.2-1B-Instruct revision `9213176726f574b556790deb65791e0c5aa438b6`; HarmBench direct prompts; frozen harmful-refusal artifacts and selected controllers.
- Sessions/groups: Original, A-LQR, and H-infinity deployment; deployed A-LQR, matched reduced-core A-LQR, and deployed H-infinity synthesis.
- Labels/targets: offline synthesis wall time, gamma-star interval width, milliseconds per generated token, and peak allocated VRAM.
- Signals/features/measures: elapsed seconds from synchronized `perf_counter`, bisection lower/upper bounds, `gamma_star`, `bisection_iterations`, generated-token count, CUDA peak allocated bytes, and resident-model baseline bytes.
- Parameters/thresholds: seed 2151; 20 synthesis trials; 100 bootstrap refits; seven inference trials; 16 prompts; batch size 8; exactly 50 new tokens; greedy decoding; KV cache off; float32 controller synthesis; bfloat16 SDPA language model.
- Outputs: `plots/computational_complexity.{pdf,png}`, `plots/synthesis_summary.csv`, and `plots/inference_summary.csv`.

## Statistics

- Tests/models: no inferential hypothesis test; timing and memory are descriptive technical replicates summarized by the median, mean, standard deviation, minimum, and maximum. Figure intervals are the interquartile range across repeated trials.
- Null hypothesis: none; this is a direct systems measurement rather than a sampled population comparison.
- Alternative hypothesis: none; the analysis does not claim statistical equivalence from seven technical replicates.
- Thresholds/decision rule: all bootstrap H-infinity fits must converge; every inference trial must generate exactly 50 tokens per prompt; methods must use identical prompts, batch size, decoding, and cache state.
- What the statistic means: milliseconds per token is synchronized wall time divided by the exact number of generated tokens; peak memory is the maximum PyTorch-allocated CUDA memory during a trial; interval width is the remaining feasible gamma bracket.
- Why this statistic is appropriate here: matched inputs and exact token counts isolate controller-associated deployment cost, while synchronized medians reduce transient timing noise without overstating generality.

## Legends

- X axis: controller method in the synthesis, latency, and memory panels; bisection iteration in the convergence panel.
- Y axis: synthesis seconds on a logarithmic scale, feasible gamma interval width on a logarithmic scale, milliseconds per generated token, or peak allocated VRAM in decimal GB.
- Color/value: black is Original, midnight blue is A-LQR, and dark red is H-infinity.
- Grouping: points show method medians; vertical intervals show the interquartile range across technical replicates.
- Ordering/sorting: Original, A-LQR, H-infinity for deployment; A-LQR then H-infinity for synthesis; iterations increase from left to right.
- Lines/markers/labels: the dark-red line is the production H-infinity bisection interval width.
- Panels: offline synthesis, gamma-star convergence, deployment latency, and deployment memory.

## Interpretation

- Deployed H-infinity synthesis takes a median 0.1245 seconds, versus 0.2015 seconds for the deployed full-state A-LQR implementation. At a matched 8-dimensional state size, H-infinity is 40.52 times slower than one A-LQR Riccati sweep because it performs repeated feasibility solves, but the absolute H-infinity cost remains about one eighth of a second.
- The production gamma search converges in 24 iterations to `gamma_star = 0.13007522`; all 100 bootstrap disturbance refits also converge in 24 iterations.
- Median latency is 5.347 ms/token for Original, 5.409 for A-LQR, and 5.459 for H-infinity. H-infinity is 0.050 ms/token above A-LQR and 2.09% above Original in this configuration.
- All three methods have the same median peak allocated VRAM of 2.882 GB at the reported precision.

## Notes

- The production pipeline synthesizes one control problem from a prompt-aggregated disturbance covariance. Bisection count is therefore not defined per prompt. The 100 bootstrap refits quantify sensitivity to the calibration records instead of treating repeated deterministic solves as independent prompts.
- The supplied action plan's proposed statement that A-LQR and H-infinity have strictly identical deployment operations is stronger than the current implementation supports: A-LQR uses a full-state semantic policy and H-infinity uses reduced-state encoding and decoding. The empirical end-to-end difference is small but nonzero.
- Run `conda run -n robust-steerability python parking/computational_complexity/computational_complexity.py --device cuda:0 --batch-size 8 --synthesis-repeats 20 --inference-repeats 7 --bootstrap-refits 100 --recompute` to repeat every measurement. Omit `--recompute` to redraw outputs from the unit cache.

## References

- `references/action_plan_computational_complexity.pdf`
- `robust_steerability/control/h_infinity.py`
- `robust_steerability/control/lqr.py`
- `robust_steerability/runtime/policy.py`
- `robust_steerability/benchmarks/harmful_runtime.py`

# bisection_stability

## Method

- Draw 100 bootstrap samples of size 50 with replacement from the frozen prompt-level calibration residual tensor.
- Refit the layer-wise disturbance covariance factor for each sample and rerun the unchanged H-infinity synthesis with the production costs and numerical tolerances.
- Plot the resulting gamma-star distribution and annotate the mean and maximum bisection iteration counts.

## Variables

- Data/input: 50 calibration residual records with shape `50 x 16 x 8`.
- Sessions/groups: 100 seeded bootstrap refits.
- Labels/targets: gamma-star and bisection iterations.
- Signals/features/measures: covariance-derived disturbance channels, feasibility result, gamma-star, bisection count, and convergence flag.
- Parameters/thresholds: seed 2151; sample size 50 with replacement; 100 refits; production search tolerance `1e-5`.
- Outputs: `plots/bisection_stability.{pdf,png}` and `cache/bisection_bootstrap_refits.csv`.

## Statistics

- Tests/models: nonparametric bootstrap distribution; no confidence interval or hypothesis test is asserted.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: every refit must return a feasible, converged controller.
- What the statistic means: dispersion in gamma-star reflects sensitivity to resampling the calibration residual set; bisection count measures numerical search work under the fixed bracket and tolerance.
- Why this statistic is appropriate here: resampling changes the empirically estimated disturbance geometry while preserving the production synthesis algorithm.

## Legends

- X axis: bootstrap gamma-star.
- Y axis: number of bootstrap refits in each bin.
- Color/value: dark red denotes H-infinity bootstrap refits.
- Grouping: one histogram over all 100 refits.
- Ordering/sorting: gamma-star increases left to right.
- Lines/markers/labels: the inset text reports mean and maximum bisection iterations.
- Panels: single panel.

## Interpretation

- Bootstrap gamma-star has mean 0.12935 and standard deviation 0.00602. All 100 refits converge in exactly 24 iterations.

## Notes

- This is calibration-set sensitivity, not prompt-level bisection timing.

## References

- `references/action_plan_computational_complexity.pdf`
- `robust_steerability/calibration/disturbances.py`

# complexity_table

## Method

- Combine the median deployed synthesis time, median deployment latency, latency change relative to Original, and median peak allocated VRAM for A-LQR and H-infinity.
- State asymptotic complexity using the dimensions actually used by the current deployed implementations.
- Export identical content as CSV, Markdown, LaTeX, and standalone PDF.

## Variables

- Data/input: `cache/synthesis_trials.csv` and `cache/inference_trials.csv`.
- Sessions/groups: A-LQR and H-infinity.
- Labels/targets: synthesis complexity, synthesis seconds, milliseconds per token, percent latency change, and peak VRAM.
- Signals/features/measures: medians across 20 synthesis trials and seven inference trials.
- Parameters/thresholds: the same protocol as `computational_complexity` above.
- Outputs: `plots/complexity_table.{csv,md,tex,pdf}`.

## Statistics

- Tests/models: descriptive medians only.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: values must be derived from complete cached trial tables.
- What the statistic means: each table cell is a direct systems measurement or asymptotic operation count.
- Why this statistic is appropriate here: the compact table directly answers the reviewer's offline-time, decoding-time, and memory questions.

## Legends

- X axis: none; this is a table.
- Y axis: none; this is a table.
- Color/value: none.
- Grouping: one row per controller.
- Ordering/sorting: A-LQR followed by H-infinity.
- Lines/markers/labels: column arrows are omitted; units appear in headers.
- Panels: single table.

## Interpretation

- H-infinity has a small absolute offline synthesis cost and the same reported peak deployment memory as A-LQR, with a measured 0.050 ms/token latency difference relative to A-LQR.

## Notes

- The A-LQR and H-infinity synthesis dimensions differ in the deployed pipeline; the table does not conceal this by assigning both methods the same symbol.

## References

- `plots/appendix_evidence.md`
- `references/action_plan_computational_complexity.pdf`

# appendix_evidence_and_reviewer_defense

## Method

- Convert the frozen numerical summaries and hardware record into an appendix-ready M.1/M.2/M.3 evidence outline.
- Map common computational-overhead critiques to measured results without modifying the manuscript.
- Retain exact versions, model revision, GPU identity, CUDA runtime, and the analysis-script checksum.

## Variables

- Data/input: `cache/summary.json`, `cache/hardware_software.json`, and `cache/run_manifest.json`.
- Sessions/groups: asymptotic synthesis, empirical synthesis, inference deployment, and reviewer-defense evidence.
- Labels/targets: numerical claims and reproducibility metadata.
- Signals/features/measures: all headline values reported by the other outputs in this unit.
- Parameters/thresholds: no additional parameters.
- Outputs: `plots/appendix_evidence.md` and `plots/reviewer_defense_matrix.{csv,md,tex,pdf}`.

## Statistics

- Tests/models: none; this output transcribes validated descriptive summaries.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: every stated number must match `cache/summary.json`.
- What the statistic means: the outline links each manuscript claim to a measured quantity and its protocol.
- Why this statistic is appropriate here: it separates empirical evidence from later manuscript wording.

## Legends

- X axis: none.
- Y axis: none.
- Color/value: none.
- Grouping: appendix sections M.1 through M.3 and reviewer critiques.
- Ordering/sorting: theory, offline runtime, deployment, then reproducibility.
- Lines/markers/labels: none.
- Panels: none.

## Interpretation

- The evidence supports describing H-infinity's offline cost as small in absolute terms and its deployment overhead as close to A-LQR, but it does not support the stronger word “identical” for latency or runtime operations.

## Notes

- Manuscript prose remains a separate writing task, as requested.

## References

- `references/action_plan_computational_complexity.pdf`
