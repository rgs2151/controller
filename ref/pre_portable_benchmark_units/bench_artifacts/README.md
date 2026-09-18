# bench_artifacts.py

## Method

- Load either the pinned TruthfulQA multiple-choice split or pinned RealToxicityPrompts, selected explicitly by `--behavior`.
- For truthfulness, sample 200 false answers, 200 true answers, and 35 true-answer Jacobian prompts. For toxicity, sample 200 toxic prompts, 200 non-toxic prompts, and 50 non-toxic Jacobian prompts.
- Load one checkpoint with its frozen A-LQR model-task configuration. Toxicity is currently provisioned only for Gemma-2-2B because that is the requested paper slice.
- Average the last-token decoder states of the false and true calibration prompts and save the resulting positive-minus-negative semantic direction and its per-depth norm.
- Compute the full last-token state Jacobian of every decoder block for each of the 35 Jacobian prompts, holding prefix states fixed. Split prompts deterministically across two GPUs and save every prompt-layer matrix.
- Average all 35 truthfulness or 50 toxicity prompt Jacobians in float64 and save the resulting controller-neutral nominal dynamics tensor in the strict artifact format shared by A-LQR and H∞.
- Write both the source-faithful A-LQR `dynamics.pt` and a value-identical strict `nominal_dynamics.pt` schema consumed directly by H∞.
- Record model-loading time, setpoint fitting time, per-shard Jacobian time, aggregation time, total elapsed time, GPU identity, peak GPU memory, software versions, command, Git state, prompt identifiers, and artifact hashes.

## Variables

- Data/input: pinned TruthfulQA multiple choice or pinned RTP train prompts.
- Sessions/groups: 200 prompts per semantic class and 35 truthfulness or 50 toxicity Jacobian prompts.
- Labels/targets: false/toxic records are undesired and true/non-toxic records are desired; the direction is desired mean minus undesired mean.
- Signals/features/measures: last-token decoder inputs, terminal decoder output, per-block full-state Jacobians, averaged dynamics, wall-clock time, and peak CUDA memory.
- Parameters/thresholds: pinned model revisions; Jacobian maximum context length 512; exact full Jacobians with memory-only VJP chunk sizes 32/16/8 and activation batch sizes 16/8/4 for Gemma/Llama/Qwen; frozen A-LQR settings Gemma λ 3, Q 0.1, R 1, Qf 0.3; Llama λ 2, Q 0.1, R 10, Qf 10; Qwen λ 3, Q 0.1, R 1, Qf 0.3.
- Outputs: cache-independent files under `cache/shared/<behavior>/<model>/`; generation-conditioned controller selections under `cache/kv_cache_<on|off>/<behavior>/<model>/<method>/`; cached-sweep diagnostics under `plots/kv_cache_<on|off>/<behavior>/<model>/`.

## Statistics

- Tests/models: direct arithmetic means of 200 prompt states per class and 35 or 50 prompt-level Jacobian matrices; descriptive wall-clock and peak-memory measurements.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: every required prompt must have one finite square Jacobian for every decoder block before either dynamics artifact is written; cache identity mismatches fail.
- What the statistic means: the semantic direction estimates the average true-versus-false residual displacement, and the averaged Jacobian estimates nominal layer-to-layer linear dynamics around the selected true prompts.
- Why this statistic is appropriate here: these are the frozen source A-LQR calibration quantities required before controller synthesis; no benchmark outcome is used for fitting.

## Legends

- X axis: for sweep diagnostics, terminal-state cost ratio $Q_f/R$.
- Y axis: for sweep diagnostics, running-state cost ratio $Q/R$.
- Color/value: mean calibration toxicity, mean calibration perplexity, or minimum feasible $\gamma^\star$; the selected controller cell has a red outline.
- Grouping: Jacobian prompts are deterministically divided into two GPU shards.
- Ordering/sorting: source record order followed by seeded sampling; decoder blocks retain model depth order.
- Lines/markers/labels: none.
- Panels: none.

## Interpretation

- `setpoint.pt` is the reusable A-LQR semantic calibration. `dynamics.pt` is the reusable controller-neutral A matrix consumed unchanged by A-LQR or H∞.
- The existing truthfulness H∞ selection is isolated under `cache/kv_cache_off/truthfulness/gemma2b/h_infinity/`; future toxicity selection is written separately.
- This unit establishes calibration only and makes no steering-performance claim.

## Notes

- Run one calibration with `python parking/bench_artifacts/bench_artifacts.py --stage all --behavior <truthfulness|toxicity> --model <model> --devices cuda:0,cuda:1`.
- Inspect one model-task pair with the same command and `--stage status`.
- `setpoint.pt` contains `contrast` and `feature_norm`. `dynamics.pt` contains the averaged tensor under `dynamics`; `dynamics.json` records the exact model, prompt texts and identifiers, Jacobian settings, implementation hashes, timing, device provenance, and artifact checksum.
- Per-shard run records retain the exact prompt identifiers corresponding to the raw Jacobian directories.
- The unit intentionally has no generation or evaluation stage. Benchmark generation and scoring are owned by `parking/bench_evaluations/`.
- Every model owns an independent artifact tree; evaluation never substitutes one model's setpoint or dynamics for another's.
- H∞ hyperparameter artifacts are indexed by cache condition, model, and task. OOD-specific calibration is not part of the current protocol.
- A valid H∞ artifact must record `D[k]D[k]ᵀ = Cov(xi[k])` on its separate calibration trajectories; source hashes prevent the old controller from being reused after this implementation change.
- `python parking/bench_artifacts/plot_toxicity_sweep.py` renders the completed Gemma-2-2B cache-off toxicity grid without rerunning generation or scoring.

## References

- `ref/2604.19018v1.pdf`.
- A-LQR paper-producing TruthfulQA calibration at upstream commit `626f757976d3b8e83bdf61a4e35bbed002b58925`.
- `robust_steerability/source_methods/calibration.py`.
- `robust_steerability/modeling/jacobians.py`.
- `ref/NON_HINFINITY_PROTOCOL.md`.
- `parking/h_infinity_hyperparameter_correction/`.
