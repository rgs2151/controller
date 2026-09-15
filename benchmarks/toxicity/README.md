# Toxicity benchmark

This benchmark owns RealToxicityPrompts in-distribution evaluation and Jigsaw
cross-dataset transfer. It uses the universal four-stage interface:

1. `artifacts` freezes fit data, the semantic setpoint, and the average of 50 Jacobians.
2. `calibrate` fits H∞ disturbance geometry and selects S-PID and H∞ on disjoint RTP development prompts.
3. `evaluate` independently generates RTP, Jigsaw, and/or MMLU responses without scoring.
4. `score` applies the scorers declared for each dataset to saved responses.

Jigsaw is a toxicity-transfer dataset and uses the toxicity classifier,
Distinct-2, and perplexity. MMLU is a capability-retention dataset and uses
deterministic A/B/C/D accuracy; neither is used for fitting or selection.
Evaluated-model KV cache defaults to off; `--kv-cache on` is retained for an
explicit comparison and uses a separate directory.

```bash
python -m robust_steerability.benchmarks.toxicity artifacts --model gemma2b --devices auto
python -m robust_steerability.benchmarks.toxicity calibrate --model gemma2b --devices auto
python -m robust_steerability.benchmarks.toxicity evaluate --model gemma2b --methods all --datasets all --devices auto
python -m robust_steerability.benchmarks.toxicity score --model gemma2b --methods all --datasets all --scorers all --devices auto
```

`benchmark.toml` is the composition surface. Adding an evaluation dataset
creates a new independent cache namespace. When RTP and Jigsaw are already
complete, extending the composition with MMLU launches only MMLU generation and
scoring; it does not load the evaluated model for the completed datasets.

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
devices, Git state, methods, datasets, cache mode, stage parameters, and the
resolved cache root. On Lightning, the complete ignored cache hierarchy is
written directly to the producing Studio's Teamspace Drive under
`~/robust-steering-cache/toxicity/<model>/`.
