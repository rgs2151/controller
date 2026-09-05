# Benchmarks

Benchmark modules own reusable prompt records, dataset loading, generation scoring, and task-level metrics. They do not synthesize controllers or register transformer hooks.

## Current Coverage

- `toxicity.py`: RealToxicityPrompts, Jigsaw, Civil Comments, and ToxicChat prompt pools plus generated-continuation toxicity probabilities.
- `mmlu.py`: Subject-level MMLU loaders and concept-shift ID/OOD prompt-set construction.

## A-LQR Parity Roadmap

- Truthfulness and TruthfulQA evaluation.
- Refusal and attack-success evaluation.
- Arbitrary concept induction and concept scoring.
- Perplexity, MMLU retention, generation diversity, and runtime measurements.

Add a benchmark module when its loader or evaluator is reused across compact units. Keep experiment-specific subsets, statistical tests, thresholds, and plots in the owning unit README and code.
