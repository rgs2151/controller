# Source-faithful comparison methods

This package is the active implementation of the non-H-infinity methods in the
A-LQR comparison. H-infinity is intentionally out of scope here.

## Frozen evaluation

- Toxicity: five repetitions of 1,000 RealToxicityPrompts generations.
- Truthfulness: five repetitions of all 817 TruthfulQA generation questions.
- Generation: temperature 1, top-p 0.3, repetition penalty 1.2; 100 new tokens
  for toxicity and 50 for truthfulness.
- `truthfulqa_true` and `truthfulqa_informative`: the exact AllenAI TruthfulQA
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
  the final selected Gemma settings. For Gemma-2-2B truthfulness, this project
  freezes conservative midpoint guesses from those grids: ITI uses 32 heads
  and alpha 10; S-PID uses lambda 1 with Kp=.7, Ki=.01, and Kd=.1.
- ITI and S-PID are never swept by this pipeline. Their saved selection files
  must equal the centralized project choices.
- GPT-2 XL is outside the original comparison table. Its added truthfulness
  rows use frozen, untuned project settings for all seven comparison methods:
  ITI (32 heads, alpha 10), ActAdd (layer 24, strength 4), AcT (strength 1 at
  the first four matched GPT-2 MLP modules), ODESteer (layer 24, time 50), and
  S-PID (lambda 1, Kp=.7, Ki=.01, Kd=.1). No method is swept on TruthfulQA.

These project choices are assumptions, not values claimed by the source paper.
All other fixed settings retain their source provenance.

## Provenance

Every stage record captures the model revision, calibration prompt IDs, selected
parameters, generation settings, scorers, software versions, requested device,
visible GPUs, and timestamps. Primary
papers and exact source revisions are catalogued in `ref/README.md` and
`ref/NON_HINFINITY_PROTOCOL.md`.
