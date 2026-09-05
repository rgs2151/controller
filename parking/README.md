# Parking

Parking contains compact units that are still being explored or iterated.

Use descriptive snake_case folder names. Do not use figure numbers here.

## Current Units

- `ood_explore/`: two-GPU frozen-LQR screen of nine 50-prompt source conditions
  plus a derived adversarial OOD condition, measuring remaining final semantic
  target error relative to each prompt's unsteered error.
- `residual_checks/`: cache-first A-LQR smoke test covering residual amplification, internal semantic tracking failure, and generated-toxicity checks under RTP-to-Jigsaw distribution shift.
- `linearization_error_explore/`: dedicated residual / zeta investigation for measuring one-step linearization mismatch, direction-aware effect, and semantic failure under shift.
