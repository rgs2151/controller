# bench_artifacts.py

## Method

- Load the pinned TruthfulQA multiple-choice validation split and format every candidate answer as `Q: <question> A: <answer>`.
- Independently sample 200 false-answer prompts, 200 true-answer prompts, and 35 true-answer prompts for Jacobian identification using seeds 42, 43, and 44.
- Load the pinned `google/gemma-2-2b` checkpoint using the frozen A-LQR TruthfulQA model-loading configuration.
- Average the last-token decoder states of the false and true calibration prompts and save the resulting positive-minus-negative semantic direction and its per-depth norm.
- Compute the full last-token state Jacobian of every decoder block for each of the 35 Jacobian prompts, holding prefix states fixed. Split prompts deterministically across two GPUs and save every prompt-layer matrix.
- Average the 35 raw prompt Jacobians in float64 and save the resulting A-LQR nominal dynamics tensor.
- Record model-loading time, setpoint fitting time, per-shard Jacobian time, aggregation time, total elapsed time, GPU identity, peak GPU memory, software versions, command, Git state, prompt identifiers, and artifact hashes.

## Variables

- Data/input: `truthful_qa`, `multiple_choice`, validation split, revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490`.
- Sessions/groups: 200 false prompts, 200 true prompts, and an independently sampled set of 35 true prompts.
- Labels/targets: false answers are undesired; true answers are desired; the target direction is desired mean minus undesired mean.
- Signals/features/measures: last-token decoder inputs, terminal decoder output, per-block full-state Jacobians, averaged dynamics, wall-clock time, and peak CUDA memory.
- Parameters/thresholds: Gemma-2-2B revision `c5ebcd40d208330abc697524c919956e692655cf`; Jacobian maximum context length 512; VJP chunk size 32; activation batch size 16; frozen A-LQR setting λ 3, Q 0.1, R 1, Qf 0.3.
- Outputs: ignored files `cache/data.json`, `cache/setpoint.pt`, `cache/jacobians/`, `cache/dynamics.pt`, `cache/timings.json`, `cache/manifest.json`, and `cache/runs/`.

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

- `setpoint.pt` and `dynamics.pt` are the reusable offline A-LQR calibration artifacts for subsequent small truthfulness/OOD steering experiments.
- This unit establishes calibration only and makes no steering-performance claim.

## Notes

- Run the complete requested calibration with `python parking/bench_artifacts/bench_artifacts.py --stage all --devices cuda:0,cuda:1`.
- Inspect paths and completion status with `python parking/bench_artifacts/bench_artifacts.py --stage status`.
- `setpoint.pt` contains `contrast` and `feature_norm`; `dynamics.pt` contains the averaged tensor under `dynamics`. Both include the complete calibration identity.
- Per-shard run records retain the exact prompt identifiers corresponding to the raw Jacobian directories.
- The unit intentionally has no generation or evaluation stage. The five-repeat benchmark remains unrun.
- Gemma-2-2B has roughly 2.6B parameters and is the frozen checkpoint intended by the user's “Gemma 3B / 2.5B” description.

## References

- `ref/2604.19018v1.pdf`.
- A-LQR paper-producing TruthfulQA calibration at upstream commit `626f757976d3b8e83bdf61a4e35bbed002b58925`.
- `robust_steerability/source_methods/calibration.py`.
- `robust_steerability/modeling/jacobians.py`.
- `ref/NON_HINFINITY_PROTOCOL.md`.
