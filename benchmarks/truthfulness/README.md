# Truthfulness benchmark

This benchmark uses the same four-stage interface as every benchmark:

1. `artifacts` freezes the fit data, semantic setpoint, and average of 35 Jacobians.
2. `calibrate` records published A-LQR parameters, fits each baseline's required
   one-time artifact, and either selects H∞ on a 50-question development set or
   synthesizes one explicitly supplied H∞ configuration without a sweep.
3. `evaluate` generates responses only; it never runs a scorer.
4. `score` applies any requested subset of independent scorers to existing responses.

The benchmark's default scorers are:

- `truthfulqa_true`: pinned TruthfulQA True judge and rubric; binary percentage.
- `truthfulqa_informative`: pinned TruthfulQA Helpful judge and rubric; binary percentage.
- `axbench_instruction_relevance`: AXBench 0–2 rubric through `gpt-4o-mini`.
- `axbench_fluency`: AXBench 0–2 rubric through `gpt-4o-mini`.

`axbench_concept_relevance` is available to benchmarks whose generation rows
contain an explicit target concept. It is not a TruthfulQA metric.

```bash
python -m robust_steerability.benchmarks.truthfulness artifacts --model llama8b --devices auto
python -m robust_steerability.benchmarks.truthfulness calibrate --model llama8b --methods alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.truthfulness evaluate --model llama8b --methods original,alqr,h_infinity --datasets id,spanish --devices auto
python -m robust_steerability.benchmarks.truthfulness score --model llama8b --methods original,alqr,h_infinity --datasets id,spanish --scorers truthfulqa_true,truthfulqa_informative,axbench_instruction_relevance,axbench_fluency --devices auto
```

To freeze an already chosen H∞ configuration without running its grid, supply
all three cost arguments during `calibrate`:

```bash
python -m robust_steerability.benchmarks.truthfulness calibrate --model gemma2b --methods h_infinity --calibration-id fixed_q0p1_qf0p31622777_r1 --h-infinity-q-over-r 0.1 --h-infinity-q-final-over-r 0.31622776601683794 --h-infinity-r 1 --devices auto
```

This still fits the task/model disturbance geometry and synthesizes H∞ once;
it skips candidate generation and judging. The resulting calibration directory
contains the exact selected gains, gamma, and Hannah diagnostic bundle.

ITI and S-PID have preserved source grids but no preserved final Gemma choice.
The project therefore freezes Gemma truthfulness best guesses from those grids:
ITI uses 32 heads and alpha 10; S-PID uses lambda 1, Kp=.7, Ki=.01, and Kd=.1.
Neither method is swept.

Evaluated-model KV cache defaults to off. `--kv-cache on` remains available for
an explicit appendix comparison and writes to a different directory. API scoring
defaults to concurrency 500 and batch size 20.

```text
cache/<model>/artifacts/
cache/<model>/datasets/
cache/<model>/calibrations/<method>/<calibration-id>/
cache/<model>/evaluations/<kv-cache-condition>/generations/<dataset>/<method>/
cache/<model>/evaluations/<kv-cache-condition>/scores/<scorer>/<dataset>/<method>/
cache/<model>/evaluations/<kv-cache-condition>/results/<dataset>/<method>.json
results/<kv-cache-condition>/<model>/<dataset>/<method>.json
```

On Lightning, this complete cache hierarchy is written directly to the
producing Studio's Teamspace Drive at
`~/robust-steering-cache/truthfulness/<model>/`. Small summaries and stage logs
are committed to Git.
