# Benchmarks

Every benchmark exposes `artifacts`, `calibrate`, `evaluate`, and `score`.
Evaluation only generates responses. Score consumes those responses through
benchmark-selected scorer definitions and never reruns the evaluated model.

Each `benchmarks/<name>/benchmark.toml` composes one base dataset with any
number of independent evaluation datasets. The base dataset owns controller
artifacts and calibration. Transfer and capability datasets reuse those frozen
objects and write separate generation, score, and result namespaces. Adding a
dataset therefore does not invalidate or regenerate completed sibling datasets.

## Current Coverage

- `toxicity.py`: RealToxicityPrompts as the base dataset, Jigsaw transfer, and
  MMLU capability retention.
- `truthfulness.py`: TruthfulQA as the base dataset, Spanish transfer, and MMLU
  capability retention.
- `robust_steerability.judges`: reusable scorer definitions with explicit model,
  rubric, score range, and backend provenance.

MMLU uses one frozen, subject-uniform sample of 1,000 five-shot test questions.
Its scorer is deterministic A/B/C/D accuracy; it does not call a judge model.

## A-LQR Parity Roadmap

- Truthfulness and TruthfulQA evaluation.
- Refusal and attack-success evaluation.
- Arbitrary concept induction and concept scoring.
- Perplexity, MMLU retention, generation diversity, and runtime measurements.

Add a dataset entry to a benchmark TOML when an existing dataset runtime and
scorer apply. Add one reusable dataset adapter or scorer only when the new
dataset introduces a genuinely new record format or metric. Keep
experiment-specific subsets, statistical tests, thresholds, and plots in the
owning unit README and code.
