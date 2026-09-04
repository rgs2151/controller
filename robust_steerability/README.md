# Robust Steerability Package

This package is the reusable boundary between language-model experiments and controller mathematics.

## Responsibilities

- `control/` receives finite-horizon tensor problems and returns controller solutions.
- `modeling/` contains all Hugging Face and transformer-hook details.
- `calibration/` converts fitted/calibration trajectories into targets, nominal dynamics, residuals, disturbance channels, and normalized coordinates.
- `runtime/` converts a controller solution or online controller into activation deltas.
- `benchmarks/` contains reusable behavior records and evaluators.

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
2. Estimate residual geometry and normalization on the calibration split.
3. Construct a `FiniteHorizonControlProblem` and synthesize a controller.
4. Freeze every fitted artifact and controller parameter.
5. Evaluate the frozen policy on held-out test conditions.

The split labels, model revision, tokenizer revision, state definition, layer mapping, and configuration hash belong in saved artifact metadata.
