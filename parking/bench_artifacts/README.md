# bench_artifacts.py

## Method

- Load the pinned TruthfulQA multiple-choice validation split and format every candidate answer as `Q: <question> A: <answer>`.
- Independently sample 200 false-answer prompts, 200 true-answer prompts, and 35 true-answer prompts for Jacobian identification using seeds 42, 43, and 44.
- Load one pinned benchmark checkpoint using its frozen A-LQR TruthfulQA model-loading configuration: Gemma-2-2B, Llama-3-8B, or Qwen-2.5-14B.
- Average the last-token decoder states of the false and true calibration prompts and save the resulting positive-minus-negative semantic direction and its per-depth norm.
- Compute the full last-token state Jacobian of every decoder block for each of the 35 Jacobian prompts, holding prefix states fixed. Split prompts deterministically across two GPUs and save every prompt-layer matrix.
- Average the 35 raw prompt Jacobians in float64 and save the resulting controller-neutral nominal dynamics tensor in the strict artifact format shared by A-LQR and H∞.
- Record model-loading time, setpoint fitting time, per-shard Jacobian time, aggregation time, total elapsed time, GPU identity, peak GPU memory, software versions, command, Git state, prompt identifiers, and artifact hashes.

## Variables

- Data/input: `truthful_qa`, `multiple_choice`, validation split, revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490`.
- Sessions/groups: 200 false prompts, 200 true prompts, and an independently sampled set of 35 true prompts.
- Labels/targets: false answers are undesired; true answers are desired; the target direction is desired mean minus undesired mean.
- Signals/features/measures: last-token decoder inputs, terminal decoder output, per-block full-state Jacobians, averaged dynamics, wall-clock time, and peak CUDA memory.
- Parameters/thresholds: pinned model revisions; Jacobian maximum context length 512; exact full Jacobians with memory-only VJP chunk sizes 32/16/8 and activation batch sizes 16/8/4 for Gemma/Llama/Qwen; frozen A-LQR settings Gemma λ 3, Q 0.1, R 1, Qf 0.3; Llama λ 2, Q 0.1, R 10, Qf 10; Qwen λ 3, Q 0.1, R 1, Qf 0.3.
- Outputs: ignored files under `cache/<model>/`: `data.json`, `setpoint.pt`, `jacobians/`, `dynamics.pt`, `timings.json`, `manifest.json`, and `runs/`.

## Statistics

- Tests/models: direct arithmetic means of 200 prompt states per class and 35 prompt-level Jacobian matrices; descriptive wall-clock and peak-memory measurements.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: all 35 prompts must have one finite square Jacobian for every decoder block before `dynamics.pt` is written; cache identity mismatches fail.
- What the statistic means: the semantic direction estimates the average true-versus-false residual displacement, and the averaged Jacobian estimates nominal layer-to-layer linear dynamics around the selected true prompts.
- Why this statistic is appropriate here: these are the frozen source A-LQR calibration quantities required before controller synthesis; no benchmark outcome is used for fitting.

## Legends

- X axis: none.
- Y axis: none.
- Color/value: none.
- Grouping: Jacobian prompts are deterministically divided into two GPU shards.
- Ordering/sorting: source record order followed by seeded sampling; decoder blocks retain model depth order.
- Lines/markers/labels: none.
- Panels: none.

## Interpretation

- `setpoint.pt` is the reusable A-LQR semantic calibration. `dynamics.pt` is the reusable controller-neutral A matrix consumed unchanged by A-LQR or H∞.
- This unit establishes calibration only and makes no steering-performance claim.

## Notes

- Run one complete calibration with `python parking/bench_artifacts/bench_artifacts.py --stage all --model <gemma2b|llama8b|qwen14b> --devices cuda:0,cuda:1`.
- Inspect one model with `python parking/bench_artifacts/bench_artifacts.py --stage status --model <model>`.
- `setpoint.pt` contains `contrast` and `feature_norm`. `dynamics.pt` contains the averaged tensor under `dynamics`; `dynamics.json` records the exact model, prompt texts and identifiers, Jacobian settings, implementation hashes, timing, device provenance, and artifact checksum.
- Per-shard run records retain the exact prompt identifiers corresponding to the raw Jacobian directories.
- The unit intentionally has no generation or evaluation stage. The five-repeat benchmark remains unrun.
- Every model owns an independent artifact tree; evaluation never substitutes one model's setpoint or dynamics for another's.

## References

- `ref/2604.19018v1.pdf`.
- A-LQR paper-producing TruthfulQA calibration at upstream commit `626f757976d3b8e83bdf61a4e35bbed002b58925`.
- `robust_steerability/source_methods/calibration.py`.
- `robust_steerability/modeling/jacobians.py`.
- `ref/NON_HINFINITY_PROTOCOL.md`.
