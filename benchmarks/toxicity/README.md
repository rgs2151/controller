# Toxicity benchmark

The current benchmark is RealToxicityPrompts (RTP) on Gemma-2-2B and
Llama-3-8B with four methods: Original, S-PID, A-LQR, and H∞. Jigsaw transfer has been removed.
MMLU remains an optional composable capability dataset but is not part of the
default run.

The benchmark uses the universal four-stage interface:

1. `artifacts` freezes 200 toxic and 200 non-toxic direction examples, the
   semantic setpoint, and the average of 50 RTP Jacobians.
2. `calibrate` writes the published A-LQR setting and a fixed S-PID setting,
   then selects H∞ on 50 disjoint RTP prompts in one pass.
3. `evaluate` generates responses without scoring.
4. `score` independently applies the requested scorers to saved responses.

H∞ calibration fixes `R=1` and evaluates 12 configurations from
`Q/R ∈ {0.01, 0.1, 1, 10}` and `Qf/R ∈ {0.01, 0.1, 0.316...}`. Each
configuration is scored for toxicity probability and percentage, Distinct-2,
perplexity, AXBench concept relevance, instruction relevance, fluency, and
AXBench overall steering. The selection rule is configured independently of
these recorded measures. Available rules are the original AXBench harmonic
mean and the default baseline-relative toxicity-quality composite. The latter
weights toxicity reduction, fluency, perplexity preservation, and Distinct-2
preservation by 0.60/0.20/0.15/0.05 and applies the quality gates recorded in
`benchmark.toml`. No other method is swept.

Passing `--selection-metric mean_axbench_overall` retains the original rule;
passing `--selection-metric toxicity_quality_composite` uses the new rule.
Changing the rule reuses completed candidate generations and scorer caches.

The final RTP evaluation uses 1,000 prompts × 5 repetitions. Its scorers are
toxicity, Distinct-2, perplexity, the three AXBench measures, and AXBench overall
steering. Evaluated-model KV cache defaults to off; `--kv-cache on` is retained
as an explicit ablation in a separate cache tree.

```bash
python -m robust_steerability.benchmarks.toxicity artifacts --model gemma2b --devices auto
python -m robust_steerability.benchmarks.toxicity calibrate --model gemma2b --devices auto
python -m robust_steerability.benchmarks.toxicity evaluate --model gemma2b --methods all --devices auto
python -m robust_steerability.benchmarks.toxicity score --model gemma2b --methods all --scorers all --devices auto
```

`benchmark.toml` is the composition surface. Adding an evaluation dataset
creates an independent cache namespace. Explicitly requesting `--datasets mmlu`
runs the frozen 200-question MMLU capability set without regenerating RTP.

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
