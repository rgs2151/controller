# Harmful-behavior benchmark

This benchmark tests whether a controller built to suppress harmful compliance
on direct HarmBench requests remains effective after the official
human-jailbreak transformation.

Stages:

1. `artifacts`: materialize a validation-only 50/50 non-exhibition versus harmful
   compliance corpus, fit the layer-wise direction and setpoint, and average 50
   nominal Jacobians into the `A` matrix shared by A-LQR and H-infinity.
2. `calibrate`: record fixed A-LQR settings and select H-infinity on 50 direct
   validation behaviors by maximum AXBench overall steering. Official validation
   ASR is retained as a tie-breaker and diagnostic.
3. `evaluate`: generate completions for all 240 standard/contextual test
   behaviors as direct requests and under five frozen official human-jailbreak
   templates, plus the frozen 200-question MMLU capability set.
4. `score`: compute test ASR with the official HarmBench test classifier;
   AXBench safe-concept relevance, instruction relevance, fluency, and their
   harmonic mean; and MMLU accuracy by deterministic exact match.

The steering target is non-exhibition/refusal, not an unrelated orthogonal
concept. The test metric is attack success rate, where lower is better. Validation
and test behaviors and classifiers remain separate.

The comparison is Original, A-LQR, and H-infinity only. The initial fast model
is `meta-llama/Llama-3.2-1B-Instruct`.

Run each stage independently:

```bash
python -m robust_steerability.benchmarks.harmful artifacts \
  --model llama32_1b_instruct --devices auto
python -m robust_steerability.benchmarks.harmful calibrate \
  --model llama32_1b_instruct --devices auto
python -m robust_steerability.benchmarks.harmful evaluate \
  --model llama32_1b_instruct --devices auto
python -m robust_steerability.benchmarks.harmful score \
  --model llama32_1b_instruct --devices auto
```

Evaluation defaults to KV-cache off. A fixed H-infinity selection can replace
the grid by passing `--h-infinity-q-over-r`,
`--h-infinity-q-final-over-r`, and `--h-infinity-r` together during calibration.

See [`docs/experiment_designs/harmful.md`](../../docs/experiment_designs/harmful.md)
for the frozen split, scoring, sample counts, and safeguards.
