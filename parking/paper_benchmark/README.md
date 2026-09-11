# paper_benchmark.py

## Method

- Load the pinned TruthfulQA validation split and format every generation question as `Q: ... A:`.
- Run the complete 817-question set five times. Each repetition contains the same questions in a separately pinned permutation and uses an independent generation seed.
- Build Gemma-2-2B A-LQR from 200 false examples, 200 true examples, and 35 independently selected true-example Jacobians, matching the paper-producing calibration protocol.
- Use the paper-selected Gemma-2-2B setting: lambda 3, Q 0.1, R 1, and terminal Q 0.3. This setting is fixed before evaluation; no A-LQR parameter sweep is run.
- Generate Original and A-LQR completions with the source sampling settings: temperature 1, top-p 0.3, repetition penalty 1.2, and at most 50 new tokens.
- Score every completion with the pinned TruthfulQA True and Info judges using the source prompts `Q: ...\nA: ...\nTrue:` and `Q: ...\nA: ...\nHelpful:`.
- Cache generation after each complete repetition and cache judge records during scoring. Cache identities include the exact calibration prompt IDs, dataset and checkpoint revisions, controller and generation parameters, seeds, and content hashes; a mismatch fails instead of reusing stale data.
- Record UTC start/end times, elapsed time, command, commit and dirty state, package versions, requested CUDA device, GPU identity, memory peaks, and output hashes for each run.

## Variables

- Data/input: `truthful_qa`, generation and multiple-choice validation configurations, revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490`.
- Sessions/groups: five stochastic repetitions of all 817 generation questions.
- Labels/targets: MC2 true answers are desired calibration examples; MC2 false answers are undesired examples.
- Signals/features/measures: last-token residual states, 35 full-state Jacobian sequences averaged into the nominal dynamics, A-LQR intervention, binary True labels, and binary Helpful labels.
- Parameters/thresholds: Gemma-2-2B revision `c5ebcd40d208330abc697524c919956e692655cf`; 200/200/35 calibration; Jacobian context limit 512; lambda 3; Q 0.1; R 1; terminal Q 0.3; calibration seed 42; judge outputs must parse exactly as `yes` or `no`.
- Outputs: ignored artifacts under `cache/`; final descriptive metrics under `cache/results/`; Markdown table under `plots/benchmark_table.md`.

## Statistics

- Tests/models: descriptive True rate, Info rate, and their marginal product T×I for each repetition; final values are the mean and standard error across five repetitions.
- Null hypothesis: none; this first run is a descriptive source-comparison check.
- Alternative hypothesis: none.
- Thresholds/decision rule: all five 817-question repetitions and all judge outputs must be complete; any malformed judge output blocks summarization.
- What the statistic means: T×I is the product of the marginal percentage of truthful answers and marginal percentage of informative answers, matching the A-LQR paper.
- Why this statistic is appropriate here: repetition-level SE represents stochastic generation variation while keeping the complete question set identical across methods.

## Legends

- X axis: none.
- Y axis: none.
- Color/value: none.
- Grouping: model and steering method.
- Ordering/sorting: fixed table order; Original first, A-LQR ninth, H∞ last.
- Lines/markers/labels: `TBD` means the cell has not been run.
- Panels: none.

## Interpretation

- The first completed comparison will show whether paper-protocol A-LQR improves Gemma-2-2B TruthfulQA T×I over the reusable Original result at full evaluation scale.
- The published controller setting is reused as a fixed prior result. The benchmark does not pay to rediscover it and does not use evaluation outcomes for parameter selection.

## Notes

- Only Gemma-2-2B, TruthfulQA ID, Original, and A-LQR are enabled in this first slice.
- Spanish, adversarial, long-context, MMLU, other methods, and other models remain unselected and unrun.
- The two GPU entry points assign Original to `cuda:0` and A-LQR to `cuda:1`.
- Prepare: `python parking/paper_benchmark/paper_benchmark.py --stage prepare`.
- Cheap validation: `python parking/paper_benchmark/paper_benchmark.py --stage smoke`.
- Two-GPU generation: `python parking/paper_benchmark/paper_benchmark.py --stage generate-pair`.
- Two-GPU judging: `python parking/paper_benchmark/paper_benchmark.py --stage score-pair`.
- Summarize each method, then render: run `--stage summarize --method original`, `--stage summarize --method alqr`, and `--stage render`.

## References

- Fixed Gemma-2-2B controller setting: `ref/lqr-activation-steering/steer/tqa_eval.py`.
- Paper-producing calibration and full evaluation: upstream commit `626f757976d3b8e83bdf61a4e35bbed002b58925`, `steer/truthfulness/tqa_data_script.py` and `lqr/supertqa.py`.
- The current refactored `*test` calibration files are debug artifacts and are not this unit's protocol.
- Paper: `ref/2604.19018v1.pdf`.

# benchmark_table.md

## Method

- Read only complete result summaries created by `paper_benchmark.py`.
- Place T×I, True, and Info into the matching model-method row; leave every uncomputed cell as `TBD`.
- Preserve the planned methods and ID/OOD/capability columns so the table can be filled one benchmark slice at a time.

## Variables

- Data/input: `cache/results/truthfulness/<model>/<method>.json`.
- Sessions/groups: model-method rows.
- Labels/targets: benchmark column names from the manuscript truthfulness table.
- Signals/features/measures: repetition mean ± repetition-level SE.
- Parameters/thresholds: only complete, hash-matched summaries are included.
- Outputs: `plots/benchmark_table.md`.

## Statistics

- Tests/models: no inferential test; cells show descriptive mean ± SE across five repetitions.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: missing results remain `TBD` and are never inferred from historical caches.
- What the statistic means: the mean is expected benchmark performance over stochastic repetitions; SE summarizes repetition-to-repetition variation.
- Why this statistic is appropriate here: every repetition evaluates the same complete question set.

## Legends

- X axis: table columns are TruthfulQA ID, future distribution shifts, True, Info, and MMLU.
- Y axis: table rows are model-method pairs.
- Color/value: none.
- Grouping: ten methods within each model.
- Ordering/sorting: Gemma-2-2B, Llama-3-8B, then Qwen-2.5-14B; fixed method order within model.
- Lines/markers/labels: arrows mark whether higher is better; `TBD` marks unrun cells.
- Panels: none.

## Interpretation

- Read Original and A-LQR within the same model block; no cross-model claim is supported until those rows are complete.
- OOD and capability cells remain unavailable until their datasets and protocols are explicitly frozen.

## Notes

- The table intentionally contains no values copied from the invalid 50-question runs.

## References

- `paper_benchmark.py` in this unit.
- `/home/dev/controller/paper/benchmark_table2.tex` for the manuscript column organization.
