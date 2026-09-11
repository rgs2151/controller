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
- `erfan_linearization_error/`: Erfan's controller-independent one-step linearization mismatch analysis.
- `erfan_model_scale_residuals/`: Erfan's residual scaling and benchmark-backed OOD sweep across model sizes.
- `erfan_ood_target_selection/`: Erfan's residual-based OOD ranking and frozen benchmark target tables.
- `erfan_toxicity_calibration/`: shared five-model reduced-state calibration and final H-infinity artifacts.
- `erfan_truthfulness_benchmark/`: 50-sample paper-aligned TruthfulQA and MMLU comparison.
- `erfan_id_toxicity_benchmark/`: 50-prompt RealToxicityPrompts comparison.
- `erfan_ood_steering_benchmark/`: 50-prompt Jigsaw, ToxicChat, and MMLU concept-shift comparison.
- `erfan_benchmark_summary/`: derived cross-benchmark summary figure.
- `paper_benchmark/`: cache-first benchmark filled one model, method, and dataset at a time; the first slice is full-set Gemma-2-2B TruthfulQA Original versus A-LQR.
- `paper_benchmark_50/`: historical pilot code and tracked summaries; its superseded local cache has been removed.
