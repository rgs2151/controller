# Robust Steerability Package

This package is the reusable boundary between language-model experiments and controller mathematics.

## Responsibilities

- `control/` owns the shared controller interface plus each controller's offline
  synthesis and online behavior.
- `modeling/` contains all Hugging Face and transformer-hook details.
- `calibration/` converts fitted/calibration trajectories into targets, actual transformer Jacobians, residuals, disturbance channels, and reduced coordinates.
- `runtime/` converts a controller solution or online controller into activation deltas.
- `benchmarks/` contains reusable behavior records and evaluators.
- `experiments/` contains the manifest runner, reduced-state calibration,
  canonical method dispatch, matched-seed generation, cache handling, and GPU
  job scheduling used by `parking/paper_benchmark_50/`.
- `experiments/diagnostics.py` exports and reads portable H∞ calibration,
  solution, and evaluation bundles. `runtime/diagnostics.py` records reduced
  online trajectories without changing the controller's feedback calculation.

Experiment-specific prompt subsets, thresholds, statistics, plots, captions, and output paths remain in their owning compact units under `parking/` or `figs/`.

## Dependency Direction

```text
parking / figs / debug
          |
          v
robust_steerability.benchmarks   robust_steerability.modeling
          |                              |
          +----------> runtime <---------+
                         |
                         v
                      control
```

`control/` must remain independent of Transformers, datasets, tokenization, prompt text, and model hooks.

## Experimental Stages

1. Fit semantic targets and nominal dynamics on the fit split.
2. Project actual full-state Jacobians into a target-preserving orthonormal basis and estimate residual geometry.
3. Construct a `FiniteHorizonControlProblem` and synthesize a controller object.
4. Freeze every fitted artifact and controller parameter.
5. Evaluate the frozen policy on held-out test conditions.

The split labels, model revision, tokenizer revision, state definition, layer mapping, and configuration hash belong in saved artifact metadata.

## H∞ diagnostic handoff

New calibrations freeze the numerical problem, residuals, coordinate maps,
performance scales, prompt splits, gains, and solver diagnostics. New H∞ benchmark
generations additionally save reduced trajectories, interventions, per-prompt
scores, text, seeds, and source/configuration hashes. Prompt checkpoints and
judge scores are reused on resume.

`load_run(path)` in `robust_steerability.experiments.diagnostics` verifies the
bundle and loads its tensors on CPU without importing Transformers or loading
an LLM. Full bundles stay in each unit's ignored cache; the runner writes a
small, tensor-free inventory into that unit's `plots/diagnostics/`.

Current runs do not whiten state coordinates. They use a target-preserving
orthonormal reduced basis, standardized performance readouts, orthonormal
control channels, normalized-depth costs, and a semantic-hyperplane reference
recomputed from the current context. Nominal feedforward is zero. Toxicity and
truthfulness have separate fit/calibration splits and targets. Hannah's solver
and diagnostic formulas are unchanged; the reference files are not edited.

All ten methods save prompt checkpoints and intervention traces. H-infinity
saves full reduced controller-coordinate tensors for the diagnostic handoff;
the other controlled methods save compact per-step norm and energy traces,
along with their controller artifacts and generated token records. Calibration
also preserves raw fit/calibration activations, attention-head states, fitted
baseline parameters, LQR/H∞ solutions, and PID settings. No old-cache
reconstruction or compatibility interface is supported. Historical Erfan outputs
remain available for inspection; fresh execution belongs to the new unit.

Run the 50-prompt matrix:
`python parking/paper_benchmark_50/paper_benchmark_50.py --devices cuda:0,cuda:1`.
A different sample count belongs to a separate analysis unit, not a smoke/full flag.

Bundle contents and independent reading/sharing:
[`parking/h_infinity_optimization/README.md`](../parking/h_infinity_optimization/README.md#diagnostic_analysis).
