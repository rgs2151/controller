# Toxicity benchmark

This unit owns the RealToxicityPrompts in-distribution benchmark and the Jigsaw cross-dataset transfer evaluation.

## Stages

1. `artifacts` freezes 200 toxic and 200 non-toxic fit prompts and computes the exact average of 50 independent non-toxic-prompt Jacobians.
2. `calibrate` fits H∞ disturbance geometry on 200 separate RTP prompts, runs the fixed Q/R–Qf/R grid on five disjoint 100-prompt development repetitions, and selects S-PID and H∞ under the frozen rule.
3. `evaluate` freezes that selection and runs five 1,000-prompt repetitions on RTP and Jigsaw. Both datasets use classifier toxicity, Dist-2, and prompt-inclusive perplexity; no MMLU calculation belongs to this benchmark.

Evaluated-model KV cache is always off. Jigsaw is never used for fitting or selection.

## Commands

```bash
python -m robust_steerability.benchmarks.toxicity artifacts --model gemma2b --devices auto
python -m robust_steerability.benchmarks.toxicity calibrate --model gemma2b --devices auto
python -m robust_steerability.benchmarks.toxicity evaluate --model gemma2b --methods all --datasets all --devices auto
```

On a larger GPU, append `--generation-batch-size <n>`. The value applies to
calibration or evaluation generation, is written into cache identity, and is
recorded in the root run manifest. Omitting it uses the frozen default of 8.

## Ownership

```text
cache/<model>/artifacts/
cache/<model>/calibrations/<method>/<calibration-id>/
cache/<model>/evaluations/kv_cache_off/
results/kv_cache_off/<model>/<dataset>/<method>.json
```

`cache/` is machine-local and ignored by Git. `results/` and root run records are tracked. Only large reusable artifacts and calibrations are explicitly published to S3.
