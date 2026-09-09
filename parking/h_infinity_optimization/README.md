# h_infinity_optimization

## Method

- Treat `ref/h_infinity.py` at SHA-256 `293c7f45a54c6ffb6caf2a13cec9b0e09396fd4921973bccc019ba09e19f96f0` as Hannah's immutable numerical oracle.
- Synthesize the oracle and optimized package controller on identical scalar, square time-varying, rectangular-channel, conditioned-cost, and deliberately infeasible finite-horizon problems on CPU and GPU.
- Preserve the same backward-game equations at every layer: `M = gamma^2 I - D' S D`, `S_bar = S + S D M^-1 D' S`, `H = R + B' S_bar B`, `K = -H^-1 B' S_bar A`, and `S = Q + A' S_bar A - (B' S_bar A)' H^-1 (B' S_bar A)`. Feasibility requires the minimum eigenvalues of both `M` and `H` to exceed `1e-7`.
- Compare feasibility decisions, `gamma_star`, feedback gains, activation interventions, and numeric diagnostics. Independently compare the scalar result with the analytic one-layer boundary and construct the exact finite-horizon disturbance-to-performance operator for every feasible small problem.
- Benchmark six-layer identity-control/identity-disturbance problems at state dimensions 64, 128, 256, 512, and 768. Run CPU measurements through dimension 128 and GPU measurements through dimension 768, synchronize CUDA around every timing, and summarize repeated runs by the median.
- The optimized package implementation preserves Hannah's float32 recursion and gamma search. It transfers problem tensors to the selected device once, omits gain storage and expensive diagnostic reductions during intermediate feasibility trials, reuses symmetric-Hessian eigenvalues for condition numbers, and collects the full result only at the selected gamma.
- Write the six-panel diagnostic to `plots/h_infinity_optimization.pdf` and `plots/h_infinity_optimization.png`, parity records to `plots/equivalence.csv`, benchmark summaries to `plots/benchmark_summary.csv`, and headline results to `plots/summary.json`.
- Record both solutions, the exact problem, and synthesis options for every synthetic parity case. These synthetic scores are not empirical model–behavior predictions.

## Variables

- Data/input: deterministic synthetic `FiniteHorizonControlProblem` tensors generated with seed 2151; no language-model data are used.
- Sessions/groups: Hannah reference versus optimized package implementation; CPU versus NVIDIA GeForce RTX 5090 on `cuda:0`.
- Labels/targets: exact oracle agreement and lower end-to-end synthesis time without changing the finite-horizon minimax solution.
- Signals/features/measures: feasibility agreement, absolute gamma error, relative Frobenius gain error, relative intervention error, relative diagnostic error, exact induced gain divided by deployed gamma, elapsed seconds, speedup, and peak CUDA memory allocated.
- Parameters/thresholds: float32 synthesis; highest float32 matmul precision; gamma interval `[0.01, 10]`; search tolerance `1e-5` for parity and `1e-4` for timing; numerical tolerance `1e-7`; deployment margin `1e-3`; 60 maximum bisection iterations; horizon 6 for timing.
- Validation-case codes: S is scalar analytic, TV is square time-varying, R is rectangular channels, C is conditioned costs, and I is capped infeasible.
- Repetitions: three timing runs through dimension 256 and two runs at dimensions 512 and 768; CPU timing is limited to dimensions 64 and 128.
- Outputs: `plots/h_infinity_optimization.{pdf,png}`, `plots/equivalence.csv`, `plots/benchmark_summary.csv`, and `plots/summary.json`.

## Statistics

- Tests/models: no inferential test; equivalence uses deterministic error thresholds and performance uses median elapsed time across repeated synchronized runs.
- Null hypothesis: none; this output is a deterministic numerical-validation and descriptive performance benchmark.
- Alternative hypothesis: none; this output is a deterministic numerical-validation and descriptive performance benchmark.
- Thresholds/decision rule: feasibility decisions must agree exactly; gamma absolute error must not exceed the configured search tolerance; relative gain and intervention errors must each be at most `5e-5`; relative diagnostic error must be at most `1e-3`; the independently computed induced gain must not exceed `gamma_used`.
- What the statistic means: error divided by tolerance below 1 passes its parity criterion; induced gain divided by `gamma_used` at or below 1 satisfies the independently evaluated attenuation bound; reference time divided by optimized time above 1 is a speedup.
- Why this statistic is appropriate here: both implementations receive identical deterministic tensors, so direct numerical agreement tests the preservation claim, while synchronized medians reduce incidental timing noise without implying population-level inference.

## Legends

- X axis: validation case for the first two panels and state dimension for the timing, speedup, and memory panels.
- Y axis: normalized equivalence error, induced-gain ratio, median synthesis seconds, reference-to-optimized speedup, or peak allocated CUDA memory in MiB.
- Color/value: dark red is Hannah's reference, midnight blue is the optimized implementation, dark green is speedup or diagnostic error, and black is gamma error or a decision boundary.
- Grouping: circles denote CPU and crosses denote GPU in the equivalence panel; timing and memory lines group by implementation.
- Ordering/sorting: validation cases follow S, TV, R, C, I; state dimensions increase from 64 to 768.
- Lines/markers/labels: dashed horizontal lines mark parity ratio 1, induced-gain ratio 1, and speedup 1.
- Panels: oracle equivalence, independent gain bound, CPU synthesis time, GPU synthesis time, GPU speedup, and GPU peak allocation.

## Interpretation

- All 10 CPU/GPU parity cases pass. Gamma-star is identical; maximum relative gain error is `2.91e-6`, maximum relative intervention error is `1.11e-5`, and maximum relative diagnostic error is `3.02e-5`.
- The analytic scalar gamma boundary differs by `3.08e-6`, within the `1e-5` search tolerance. The largest independently computed induced-gain ratio is `0.999129`, below the deployed gamma bound.
- At the current 768-state, six-layer scale, median GPU synthesis decreases from about `3.63 s` to `0.95 s`, a `3.8x` speedup. Peak CUDA allocation decreases from `154.88 MiB` to `118.38 MiB`.

## Notes

- Run `python parking/h_infinity_optimization/h_infinity_optimization.py --device cuda:0 --repeats 3 --dimensions 64 128 256 512 768 --recompute` to repeat synthesis and timing. Omit `--recompute` to regenerate tables and plots from the matching local cache.
- GPU timings and memory are hardware- and software-stack-specific; mathematical parity is the portable result.
- The reference checksum is checked before every run. The analysis stops if Hannah's preserved file changes.
- Diagnostic protocol 2 also saves `cache/parity_solutions/<device>/<case>/synthesis.pt`, `cache/benchmark_summary.csv`, and `cache/summary.json`. Existing protocol-1 timing caches and plotted results were not regenerated when this export was integrated; protocol 2 requires an explicitly requested recomputation.

## References

- `ref/h_infinity.py`
- `robust_steerability/control/h_infinity.py`
- `robust_steerability/control/metrics.py`

# diagnostic_analysis

## Method

- Use the supplied finite-horizon matrices and disjoint fit/calibration prompt IDs to compute H∞ feasibility, the numerical gamma boundary, and `S_rob = 1 / gamma_star`. Reuse the deployed solution when exporting a benchmark calibration.
- Preserve the reduced fit/calibration states, residuals, PCA basis, target-preserving bases, semantic targets, references, and disturbance construction. Compare empirical residual covariance with `D Dᵀ` using relative Frobenius error.
- Attach held-out prompt/seed observations only after the score was frozen. Retain raw evaluator scores and an explicit binary-success definition; average seeds within each prompt, then prompts within a run.
- For prospective, fully normalized model–behavior pairs, prepare predictor/reliability rows, Spearman correlations, descriptive regression lines, and grouped held-out predictions. Missing baseline predictors remain missing; they are not replaced with invented values.
- Export numerical panel inputs only; this script does not redraw or modify the frozen figure sketch.

## Variables

- Input: Hannah's `problem`, `record`, `splits`, `normalization`, `calibration`, `predictors`, and optional `options` bundle. All tensor files contain ordinary dictionaries and CPU tensors, readable with `torch.load(..., map_location="cpu", weights_only=True)`.
- A frozen run contains `calibration_input.pt`, `controller.pt`, `score.json`, `manifest.json`, and controller/exporter source snapshots. Benchmark copies additionally contain `benchmark.json`, runtime source snapshots, `online/<subset>/` prompt checkpoints/traces, and `evaluations/<subset>/` observations, summaries, and checksums.
- Offline tensors: `A/B/D/Q/R/Q_final`, gains, reduced states, residuals, full layer-wise encoders/decoders/means, PCA bases, fit labels, semantic readouts, setpoints, residual covariance, and retained disturbance ranks. Raw fit/calibration hidden states and attention-head outputs are retained, along with exact fit/calibration records and source snapshots.
- Online tensors: call-ordered layer index, reduced state `x`, full-state feedback `x - reference`, total control `u`, deviation control, reduced intervention `Bu`, and hidden-delta squared norm. A trace covers one prompt's prefill and decoding; it is not a collection of independent depth trajectories. `control_energy` is the unweighted sum of `u²` over recorded calls/layers/channels. Hidden-delta energy is measured after dtype casting, before addition rounding.
- Current benchmark state coordinates use a target-preserving orthonormal reduced basis and are not whitened. Control channels are orthonormal, performance readouts use measured calibration scales, and running state and control costs are depth-weighted. The reduced basis and nominal ridge dynamics are fit-only. Protected readouts span the orthogonal complement of the target in the reduced representation; they are not independently trained behavioral probes.
- Runtime recomputes the nearest semantic-hyperplane reference from the current context at each forward call. Nominal feedforward is zero, so the applied intervention is controller feedback decoded through the next-layer basis. Toxicity and truthfulness use separately fitted targets and disjoint behavior-specific calibration records.
- Toxicity success: classifier probability `< 0.5`. TruthfulQA success: both judges say yes for a completion. MMLU success: parsed answer equals the answer index. These diagnostics do not change the existing aggregate table metrics.
- Predictors: log parameter count, probe accuracy, semantic SNR, linearization error, Jacobian-subspace similarity, Gramian metric, nominal LQR objective, minimum nominal control energy, `S_rob`, and `-log(gamma_star)`. Only the parameter-count and gamma-derived predictors are automatically populated.
- Analysis outputs under the chosen `cache_root/analyses/<analysis_id>/`: `pair_table.json`, `excluded_runs.json`, `cv_predictions.json`, `cv_metrics.json`, `correlations.json`, `descriptive_lines.json`, `unavailable_analyses.json`, and `panel_manifest.json`.

## Statistics

- Models: one predictor at a time in ordinary least squares with an intercept; the response is prompt-averaged binary success for a model–behavior pair. Predictor standardization uses the training fold only. Constant training predictors reduce to a training-mean model.
- Validation: leave-model, leave-family, leave-behavior, and leave-scale out. At least three training pairs are needed per fold; incomplete coverage produces no aggregate predictive metric.
- Correlations: Spearman rho with 500 model-cluster bootstrap samples, seed 2151; report percentile 95% intervals only with at least 100 nondegenerate resamples.
- Null/alternative: no hypothesis test or p-value is computed. Correlation zero means no monotonic association; positive held-out `R²` means improvement over the pooled observed-response mean, not a significance claim.
- Eligibility: require a converged, feasible, finite positive gamma and nonzero disturbance channel; exactly one matching evaluation per pair; the same declared calibration/normalization protocols and controller-matching rules; fully normalized coordinates; and evaluation timestamps later than the frozen score.
- Interpretation: higher `S_rob` predicts greater robustness under the declared coordinates. The descriptive line is in-sample; grouped predictions test transfer to unseen groups. These exports implement Hannah's univariate analysis, not the paper's full mixed-effects model.

## Legends

- No rendered figure, axes, colors, or markers are produced by this script.
- Panel A inputs describe the frozen calibration; B contains predictor versus held-out reliability; C contains leave-model-out metrics; D contains leave-family-out predictions. These are Hannah's export-panel labels, not a remapping of the frozen sketch.
- Rows are sorted by run/evaluation paths. `excluded_runs` gives the reason a run is not in the prospective normalized cohort.

## Interpretation

- New benchmarks record the full handoff during calibration and generation. Existing historical outputs are preserved but are not imported into the fresh benchmark tables.
- The full reference-mandated bundle is recorded. Only automatically defined predictors are populated; additional empirical predictor analyses still need their own calculations. Missing predictor values remain explicit.

## Notes

- Reference retained unchanged: `ref/h_infinity_optimization.py`, SHA-256 `6d75d6c734e647322493d69a59798c3c98fd2213bf929dbfc8c50e1c2ce55ce4`. Its numerical exporter and panel preparation were copied into the package/this unit; the optimized H∞ solver equations were not changed.
- Targeted tests cover schema validation, exact exported numerical values, leakage/timestamp checks, corrupted-file rejection, independent CPU reading after ZIP transfer, recording parity, prompt resume, and all ten methods. GPU recording parity was checked separately. Do not run the whole test suite merely to inspect a bundle.
- Fresh run: `python parking/paper_benchmark_50/paper_benchmark_50.py --devices cuda:0,cuda:1`. Each evaluation condition contains 50 prompts. Failed/incomplete runs resume only matching current checkpoints; there is no old-cache reconstruction command.
- Calibration bundles live in `parking/paper_benchmark_50/cache/controllers/<toxicity|truthfulness>/<model>_diagnostics/runs/<run_id>/`. Each benchmark copies them into `cache/jobs/<toxicity|truthfulness>/<model>/diagnostics/runs/<run_id>/`; shared calibration inputs remain immutable.
- Hannah can set `RUN_DIR` to a received run folder and read it without an LLM:

  ```python
  from pathlib import Path
  from robust_steerability.experiments.diagnostics import load_run
  import torch

  run = Path("/path/to/received/calibration-RUN_ID")
  data = load_run(run)  # checks hashes; CPU tensors; no Transformers import
  problem = data["calibration"]["problem"]
  residuals = data["calibration"]["calibration"]["residuals"]
  gains = data["controller"]["gains"]
  observations = next(iter(data["evaluations"].values()))["observations"]
  trace = torch.load(run / observations[0]["trace_file"], map_location="cpu", weights_only=True)
  ```

- CLI inspection: `python parking/h_infinity_optimization/diagnostic_analysis.py inspect --run "$RUN_DIR"`. Packing: `python parking/h_infinity_optimization/diagnostic_analysis.py pack --run "$RUN_DIR" --output tmp/hannah_diagnostics.zip`. ZIP creation never uploads anything or overwrites an existing archive.
- Git-friendly reports are written to each benchmark's `plots/diagnostics/<toxicity|truthfulness>_<model>.json`: score, provenance, evaluation summaries, per-file byte counts, and hashes; no prompts or tensor contents. Full tensor/trace bundles remain ignored and can be shared privately through Drive. Sharing a full bundle includes benchmark prompt/completion text; review that before publishing.
- To use Hannah's panel command, unpack selected model bundles under a chosen `cache_root/runs/`, then call `prepare-panels --cache-root <cache_root> --analysis-id <id> --controller hinf --shift <subset> --protocol-id <protocol> --normalization-id <normalization>`. Run IDs are calibration-specific; keep different benchmark exports with the same run ID in separate roots unless deliberately combining their disjoint evaluation folders. Existing analysis IDs are immutable. Raw-coordinate runs are not included in normalized comparisons.

## References

- `ref/h_infinity_optimization.py`
- `robust_steerability/experiments/diagnostics.py`
- `robust_steerability/experiments/calibration.py`
- `robust_steerability/experiments/runner.py`
- `robust_steerability/runtime/diagnostics.py`
- `tests/test_diagnostics.py`
