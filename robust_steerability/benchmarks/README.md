# Benchmarks

Every benchmark exposes `artifacts`, `calibrate`, `evaluate`, and `score`.
Evaluation only generates responses. Score consumes those responses through
benchmark-selected scorer definitions and never reruns the evaluated model.

## Current Coverage

- `toxicity.py`: RealToxicityPrompts, Jigsaw, Civil Comments, and ToxicChat prompt pools plus generated-continuation toxicity probabilities.
- `truthfulness.py`: separate artifact, calibration, generation-only evaluation, and independent scoring stages for TruthfulQA ID and Spanish.
- `robust_steerability.judges`: reusable scorer definitions with explicit model,
  rubric, score range, and backend provenance.

## A-LQR Parity Roadmap

- Truthfulness and TruthfulQA evaluation.
- Refusal and attack-success evaluation.
- Arbitrary concept induction and concept scoring.
- Perplexity, MMLU retention, generation diversity, and runtime measurements.

Add a benchmark module when its loader or evaluator is reused across compact units. Keep experiment-specific subsets, statistical tests, thresholds, and plots in the owning unit README and code.
