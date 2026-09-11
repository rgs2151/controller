# Source-faithful comparison methods

This package is the active implementation of the non-H-infinity methods in the
A-LQR comparison. H-infinity is intentionally out of scope here.

## Frozen evaluation

- Toxicity: five repetitions of 1,000 RealToxicityPrompts generations.
- Truthfulness: five repetitions of all 817 TruthfulQA generation questions.
- Generation: temperature 1, top-p 0.3, repetition penalty 1.2; 100 new tokens
  for toxicity and 50 for truthfulness.
- Truthfulness judges: the exact AllenAI TruthfulQA truth and information
  checkpoints and their `Q:/A:/True:` and `Q:/A:/Helpful:` rubrics.
- MMLU: 5-shot. Toxicity PPL: Mistral-7B, as specified by the A-LQR paper.

There is no 50-prompt final protocol. A non-full evaluation count is rejected
before model loading.

## One-time calibration

| Method | Toxicity | Truthfulness | Jacobians |
|---|---:|---:|---:|
| A-LQR | 200/200 | 200/200 | 50 / 35 |
| S-PID | 200/200 | 200/200 | none |
| ActAdd | 100/100 | 100/100 | none |
| ITI | 80/80 | 80/80 | none |
| Mean/Linear/PID-AcT | 200/200 | 400/400 | none |
| ODESteer | 5000/5000 | 1800/1800 | none |

A-LQR and S-PID share the same semantic-setpoint artifact. Only A-LQR creates
the dynamics/Jacobian artifact.

## Parameter selection

The final runner always evaluates one configuration. It never sweeps candidates
on the final benchmark.

- A-LQR, ActAdd, Mean/Linear/PID-AcT, and supported ODESteer rows use fixed
  settings preserved by the paper-producing repository.
- The preserved A-LQR repository contains ITI and S-PID candidate grids but not
  the final selected Gemma settings. These two methods therefore require an
  explicit, recorded development-set selection from those exact grids. The
  runner fails if that selection is absent or outside the grid.

This is deliberate: the package neither invents missing values nor repeats a
full grid over the final evaluation set.

## Provenance

Every manifest is method-specific and records the model revision, implementation
hashes, calibration prompt IDs, selected parameters, generation settings,
scorers, software versions, device, GPU, timestamps, and output hashes. Primary
papers and exact source revisions are catalogued in `ref/README.md` and
`ref/NON_HINFINITY_PROTOCOL.md`.
