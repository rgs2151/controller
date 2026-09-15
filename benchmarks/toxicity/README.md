# Toxicity benchmark

This benchmark owns RealToxicityPrompts in-distribution evaluation and Jigsaw
cross-dataset transfer. It uses the universal four-stage interface:

1. `artifacts` freezes fit data, the semantic setpoint, and the average of 50 Jacobians.
2. `calibrate` fits H∞ disturbance geometry and selects S-PID and H∞ on disjoint RTP development prompts.
3. `evaluate` generates five 1,000-prompt repetitions on RTP and/or Jigsaw without scoring.
4. `score` applies any selected subset of `toxicity_classifier`, `distinct_2`, and `perplexity` to saved responses.

MMLU does not belong to this benchmark. Jigsaw is never used for fitting or
selection. Evaluated-model KV cache defaults to off; `--kv-cache on` is retained
for an explicit comparison and uses a separate directory.

```bash
python -m robust_steerability.benchmarks.toxicity artifacts --model gemma2b --devices auto
python -m robust_steerability.benchmarks.toxicity calibrate --model gemma2b --devices auto
python -m robust_steerability.benchmarks.toxicity evaluate --model gemma2b --methods all --datasets all --devices auto
python -m robust_steerability.benchmarks.toxicity score --model gemma2b --methods all --datasets all --scorers all --devices auto
```

```text
cache/<model>/artifacts/
cache/<model>/datasets/
cache/<model>/calibrations/<method>/<calibration-id>/
cache/<model>/evaluations/<kv-cache-condition>/generations/<dataset>/<method>/
cache/<model>/evaluations/<kv-cache-condition>/scores/<scorer>/<dataset>/<method>/
cache/<model>/evaluations/<kv-cache-condition>/results/<dataset>/<method>.json
results/<kv-cache-condition>/<model>/<dataset>/<method>.json
```

Stage logs record start, finish, elapsed time, host, visible GPUs, requested
devices, Git state, methods, datasets, cache mode, and stage parameters. Only
large artifacts and calibrations are transported through S3.
