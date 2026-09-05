# Parking

Parking contains compact units that are still being explored or iterated.

Use descriptive snake_case folder names. Do not use figure numbers here.

## Current Units

- `ood_adversarial/`: iterative two-GPU adversarial prompt search against the
  frozen A-LQR controller, comparing 50 paired prompts across named attack
  attempts by both median steering degradation and cross-prompt spread.
- `ood_explore/`: two-GPU frozen-LQR screen of nine 50-prompt source conditions
  plus a derived adversarial OOD condition, measuring remaining final semantic
  target error relative to each prompt's unsteered error.
- `residual_checks/`: cache-first A-LQR smoke test covering residual amplification, internal semantic tracking failure, and generated-toxicity checks under RTP-to-Jigsaw distribution shift.
