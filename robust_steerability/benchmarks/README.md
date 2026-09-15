# Benchmarks

Every benchmark exposes `artifacts`, `calibrate`, `evaluate`, and `score`.
Evaluation only generates responses. Score consumes those responses through
benchmark-selected scorer definitions and never reruns the evaluated model.

Each `benchmarks/<name>/benchmark.toml` composes one base dataset with any
number of independent evaluation datasets. The base dataset owns controller
artifacts and calibration. Transfer and capability datasets reuse those frozen
objects and write separate generation, score, and result namespaces. Adding a
dataset therefore does not invalidate or regenerate completed sibling datasets.

The same composition owns `available_methods` and `default_methods`. Benchmark
drivers validate those keys through the shared method registry; they do not own
private method lists. Adding a method means implementing its self-contained
fit/policy adapter once, registering its orchestration metadata once, and then
adding its key to the benchmark TOML files where it is scientifically applicable.
No dataset adapter or scorer changes, and completed methods remain untouched.

## Current Coverage

- `toxicity.py`: RealToxicityPrompts as the base dataset, Jigsaw transfer, and
  MMLU capability retention.
- `truthfulness.py`: TruthfulQA as the base dataset, Spanish transfer, and MMLU
  capability retention.
- `lciteeval.py`: AXBench concept-499 steering transferred unchanged across the
  40 matched L-CiteEval HotpotQA cases at approximately 8K, 16K, and 32K.
- `mgsm.py`: one paired English-to-Spanish MGSM direction transferred across
  nine held-out input languages while preserving exact numerical accuracy.
- `harmful.py`: HarmBench non-exhibition/refusal steering evaluated on direct
  requests, five official human-jailbreak cases per behavior, and MMLU
  capability retention.
- `robust_steerability.judges`: reusable scorer definitions with explicit model,
  rubric, score range, and backend provenance.
- `judges/exact.py`: deterministic MMLU, L-CiteEval answer-overlap, and AXBench
  aggregate calculations.
- `judges/lciteeval.py`: the released citation precision/recall/F1 procedure and
  pinned local DeBERTa NLI loader.
- `judges/mgsm.py`: language-independent final-number extraction and the exact
  AXBench deterministic Spanish-language rule.
- `judges/openai.py`: batched concurrent execution of the three independent
  AXBench 0–2 rubrics.

MMLU uses one frozen, subject-uniform sample of 200 five-shot test questions.
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
