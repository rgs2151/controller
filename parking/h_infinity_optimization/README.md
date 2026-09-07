# h_infinity_optimization

## Method

- Treat `ref/h_infinity.py` at SHA-256 `293c7f45a54c6ffb6caf2a13cec9b0e09396fd4921973bccc019ba09e19f96f0` as Hannah's immutable numerical oracle.
- Synthesize the oracle and optimized package controller on identical scalar, square time-varying, rectangular-channel, conditioned-cost, and deliberately infeasible finite-horizon problems on CPU and GPU.
- Preserve the same backward-game equations at every layer: `M = gamma^2 I - D' S D`, `S_bar = S + S D M^-1 D' S`, `H = R + B' S_bar B`, `K = -H^-1 B' S_bar A`, and `S = Q + A' S_bar A - (B' S_bar A)' H^-1 (B' S_bar A)`. Feasibility requires the minimum eigenvalues of both `M` and `H` to exceed `1e-7`.
- Compare feasibility decisions, `gamma_star`, feedback gains, activation interventions, and numeric diagnostics. Independently compare the scalar result with the analytic one-layer boundary and construct the exact finite-horizon disturbance-to-performance operator for every feasible small problem.
- Benchmark six-layer identity-control/identity-disturbance problems at state dimensions 64, 128, 256, 512, and 768. Run CPU measurements through dimension 128 and GPU measurements through dimension 768, synchronize CUDA around every timing, and summarize repeated runs by the median.
- The optimized package implementation preserves Hannah's float32 recursion and gamma search. It transfers problem tensors to the selected device once, omits gain storage and expensive diagnostic reductions during intermediate feasibility trials, reuses symmetric-Hessian eigenvalues for condition numbers, and collects the full result only at the selected gamma.
- Write the six-panel diagnostic to `plots/h_infinity_optimization.pdf` and `plots/h_infinity_optimization.png`, parity records to `plots/equivalence.csv`, benchmark summaries to `plots/benchmark_summary.csv`, and headline results to `plots/summary.json`.

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

## References

- `ref/h_infinity.py`
- `robust_steerability/control/h_infinity.py`
- `robust_steerability/control/metrics.py`
