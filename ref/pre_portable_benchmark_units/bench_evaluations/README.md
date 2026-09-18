# bench_evaluations.py

## Method

- Evaluate one frozen model, method, behavior, and distribution at a time.
- Require an explicit evaluated-model KV-cache condition and isolate every generation, score, log, run record, and summary under that condition.
- Reuse the model-task setpoint and averaged Jacobian dynamics from `parking/bench_artifacts/cache/shared/`; this unit does not refit them.
- Score TruthfulQA with the pinned True and Helpful judges and score RTP with the pinned toxicity, Dist-2, MMLU, and perplexity procedures.
- Write method summaries for synchronized rendering by `figs/bench_table/`.

## Variables

- Data/input: pinned TruthfulQA, Spanish TruthfulQA, RTP, and MMLU records.
- Sessions/groups: five 817-question TruthfulQA repetitions or five 1,000-prompt RTP repetitions.
- Labels/targets: true versus false answers for truthfulness; non-toxic versus toxic RTP prompts for toxicity.
- Signals/features/measures: generated continuations, T×I, True, Info, classifier toxicity, Dist-2, MMLU accuracy, and perplexity.
- Parameters/thresholds: explicit `--kv-cache on|off`; evaluated-model cache state applies to both behavior and MMLU generation, while judges may cache.
- Outputs: `cache/kv_cache_<on|off>/{data,generations,scores,results,run_records,logs}/`.

## Statistics

- Tests/models: descriptive repetition means and standard errors; MMLU uses prompt-level Bernoulli standard error.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: all five repetitions must be complete; toxic probability greater than 0.5 is toxic; judge parsing follows the pinned source rubric.
- What the statistic means: the standard error describes stochastic-generation variability across matched repetitions.
- Why this statistic is appropriate here: every method receives the same frozen questions, generation settings, and cache condition.

## Legends

- X axis: none.
- Y axis: none.
- Color/value: none.
- Grouping: KV-cache condition, behavior, distribution, model, then method.
- Ordering/sorting: fixed prompt order inside each frozen repetition.
- Lines/markers/labels: `TBD` is added only by the table unit for absent summaries.
- Panels: none.

## Interpretation

- Cache-on and cache-off values are separate experimental conditions and must never be combined.
- The preserved existing full-run summaries are under `cache/kv_cache_on/`; new controlled runs use the separate cache-off tree.

## Notes

- Every command requires `--kv-cache on` or `--kv-cache off`.
- This general runner remains available for TruthfulQA and its Spanish shift. The frozen cache-off toxicity workflow below owns the new RTP-to-Jigsaw run.

## References

- `parking/bench_artifacts/`.
- `figs/bench_table/`.
- `ref/2604.19018v1.pdf`.

# toxicity_kv_cache_off.py

## Method

- Build Gemma-2-2B toxicity artifacts from 200 toxic and 200 non-toxic RTP prompts and 50 non-toxic RTP Jacobian prompts.
- Reuse the resulting controller-neutral A matrix for A-LQR and H∞. Fit H∞ disturbance geometry on 200 separate RTP prompts.
- Calibrate on five disjoint 100-prompt RTP development sets. Select S-PID λ from the preserved source grid and H∞ from the frozen 8-by-4 Q/R–Qf/R grid.
- Select the lowest mean development toxicity whose mean perplexity is no more than 1.5 times Original; resolve ties by lower perplexity and then the smaller controller setting.
- Freeze the selected controller before test generation. Never recalibrate on Jigsaw.
- Evaluate Original, S-PID, A-LQR, and H∞ on five 1,000-prompt RTP repetitions and five 1,000-prompt Jigsaw repetitions with evaluated-model KV cache off.
- Score RTP and Jigsaw continuations symmetrically with classifier toxicity, Dist-2, and prompt-inclusive perplexity.

## Variables

- Data/input: RTP revision `f216297...`, Jigsaw revision `2bf801d...`, and Gemma-2-2B revision `c5ebcd4...`.
- Sessions/groups: 200/200 semantic-fit prompts, 50 Jacobian prompts, 200 H∞ disturbance prompts, five × 100 tuning prompts, and five × 1,000 final prompts per distribution.
- Labels/targets: toxic RTP prompts are undesired; non-toxic RTP prompts are desired; Jigsaw has no fitting role.
- Signals/features/measures: last-token setpoints and Jacobians, rank-8 disturbance residuals, H∞ gains and γ*, continuation toxicity, Dist-2, and perplexity.
- Parameters/thresholds: A-LQR uses the published Gemma toxicity setting; S-PID uses fixed gains `(0.7, 0.01, 0.1)` and chooses λ from `{0.5, 1}`; H∞ fixes λ=3.5 and R=1, sweeps Q/R over eight half-decades from 0.01 to 31.6, and Qf/R over four half-decades from 0.01 to 0.316; evaluated-model KV cache is off.
- Outputs: cache-independent tensors under `parking/bench_artifacts/cache/shared/toxicity/gemma2b/`; cache-conditioned H∞/S-PID selections under `parking/bench_artifacts/cache/kv_cache_off/toxicity/gemma2b/`; final outputs under this unit's `cache/kv_cache_off/`.

## Statistics

- Tests/models: descriptive means and standard errors across five final repetitions; no inferential test.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: toxic-class probability greater than 0.5 is toxic; calibration candidates must satisfy mean PPL ≤ 1.5 × Original calibration PPL.
- What the statistic means: RTP measures in-distribution toxicity control and output quality; Jigsaw measures cross-dataset transfer with the same toxicity and quality metrics.
- Why this statistic is appropriate here: identical toxicity, lexical-diversity, and perplexity procedures compare the two prompt populations while all fitting remains RTP-only.

## Legends

- X axis: none.
- Y axis: none.
- Color/value: none.
- Grouping: distribution and method.
- Ordering/sorting: Original, S-PID, A-LQR, H∞.
- Lines/markers/labels: table arrows indicate the preferred direction; absent results are `TBD`.
- Panels: RTP is the ID metric block and Jigsaw is the OOD metric block.

## Interpretation

- The Jigsaw column asks whether a controller fitted and selected only on RTP transfers to a different toxicity-text distribution.
- Jigsaw is evaluated for toxicity, not truthfulness, and is never used to tune the controller.

## Notes

- No stage has been executed in the new cache-off toxicity tree.
- GPU placement is machine-agnostic: pass any comma-separated CUDA devices through `--devices`, for example `cuda:0,cuda:1,cuda:2,cuda:3`.
- Full future command: `python parking/bench_evaluations/toxicity_kv_cache_off.py --stage all --devices cuda:0,cuda:1`.
- Staged commands are available for artifact creation, calibration, final generation, scoring, and summarization so work can resume from cached completions.

## References

- `parking/bench_artifacts/bench_artifacts.py`.
- `robust_steerability/experiments/calibration.py`.
- `robust_steerability/source_methods/protocol.py`.
- `ref/lqr-activation-steering/steer/toxicity/`.
