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
- `toxicity_calibration_sweep/`: descriptive toxicity, perplexity, and `gamma_star` heatmaps for the completed Gemma-2-2B H∞ calibration grid.
- `lciteeval_explore/`: four-panel behavioral heatmap of the completed Qwen2.5-3B-Instruct L-CiteEval H∞ calibration grid.
- `truthfulqa_spanish/`: frozen 817-question Spanish TruthfulQA translation, its reproducible construction pipeline, and translation-fidelity audit.
- `kv_cache_investigation/`: matched 100-prompt Spanish TruthfulQA comparison of generation with and without the transformer KV cache for Original, A-LQR, and H∞.
- `truthful_kv_off_manual/`: matched 50-prompt ID/Spanish TruthfulQA comparison with generation KV cache off and a manually specified H∞ cost setting.
- `h_infinity_hyperparameter_correction/`: Gemma-2-2B TruthfulQA H∞ calibration
  over a 32-cell Q/R–Qf/R grid, with λ fixed at 3 and each cell averaged over
  five generations of the same 100 questions. The corrected covariance-factor
  run selects Q/R=0.1 and Qf/R=0.1 and registers that controller with the
  benchmark-artifact unit.
