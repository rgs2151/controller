# Truthfulness benchmark

This benchmark uses the same four-stage interface as every benchmark:

1. `artifacts` freezes the fit data, semantic setpoint, and average of 35 Jacobians.
2. `calibrate` records published A-LQR parameters and selects H∞ parameters on the fixed development set.
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

Large artifacts and calibrations may be published to the equivalent S3 prefix.
Generations and score outputs remain on the machine that ran them; small summaries
and stage logs are committed to Git.
