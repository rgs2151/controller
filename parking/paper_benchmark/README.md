# paper_benchmark.py

## Method

- Prepare separate immutable caches for TruthfulQA and RealToxicityPrompts (RTP).
- For TruthfulQA, evaluate all 817 generation questions in five seeded permutations and fit Gemma-2-2B A-LQR from 200 false answers, 200 true answers, and 35 independently sampled true-answer Jacobians.
- For toxicity, independently sample 1,000 prompts from all scored RTP prompts in each of five repetitions. Fit A-LQR from 200 prompts with toxicity at least 0.8, 200 prompts with toxicity at most 0.1, and 50 independently sampled non-toxic Jacobians.
- Use the published Gemma settings without an evaluation-time sweep: TruthfulQA uses λ 3, Q 0.1, R 1, Qf 0.3; toxicity uses λ 3.5, Q 0.1, R 1, Qf 0.1.
- Generate with temperature 1, top-p 0.3, repetition penalty 1.2, and at most 50 new tokens for TruthfulQA or 100 for toxicity.
- Preserve the source cache behavior: Original and LFS-controlled RTP decoding disable KV caching; Original MMLU enables it and LFS-controlled MMLU disables it.
- Score TruthfulQA with the pinned True and Helpful judges. Score toxicity with the pinned RoBERTa classifier, corpus-level Dist-1/2/3, and prompt-inclusive Mistral-7B perplexity truncated to 128 tokens.
- Evaluate toxicity-steered capability on one shared, seeded set of 1,000 five-shot MMLU questions using one greedy answer token.
- Cache calibration, generation, scoring, provenance, prompt identities, and exact dependency revisions. Any identity mismatch fails.

## Variables

- Data/input: TruthfulQA revision `741b827...`; RTP revision `f216297...`; MMLU revision `c30699e...`.
- Sessions/groups: five 817-question TruthfulQA repetitions; five 1,000-prompt RTP repetitions; one 1,000-question MMLU set.
- Labels/targets: true versus false MC2 answers; non-toxic RTP prompts are desired and toxic RTP prompts are undesired; MMLU answer indices A–D.
- Signals/features/measures: last-token residual states, full-state Jacobians, continuations, True/Helpful labels, toxic labels, distinct n-grams, sequence perplexity, and MMLU correctness.
- Parameters/thresholds: Gemma-2-2B revision `c5ebcd4...`; toxicity classifier revision `048c25b...`; Mistral-7B revision `27d67f1...`; seed 42.
- Outputs: ignored artifacts under `cache/data/`, `cache/calibrations/`, `cache/generations/`, `cache/scores/`, `cache/results/`, and `cache/run_records/`.

## Statistics

- Tests/models: descriptive repetition means and standard errors; no inferential hypothesis test.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: all five repetitions must be complete; toxic probability greater than 0.5 is toxic; judge answers must parse exactly as yes/no; MMLU must parse exactly as A, B, C, or D.
- What the statistic means: repetition-level SE measures stochastic-generation variability; MMLU uses prompt-level Bernoulli SE on its single shared set.
- Why this statistic is appropriate here: every method receives the same pinned protocol and prompt identities, while the five generative repetitions reproduce the source benchmark design.

## Legends

- X axis: none.
- Y axis: none.
- Color/value: none.
- Grouping: behavior, model, and steering method.
- Ordering/sorting: Original first, A-LQR ninth, H∞ last in output tables.
- Lines/markers/labels: `TBD` means unrun.
- Panels: none.

## Interpretation

- The first slices compare Original and paper-protocol A-LQR on Gemma-2-2B before other methods or models are added.
- A-LQR settings are fixed before final evaluation; the reported test prompts are never used for parameter selection.

## Notes

- The source RTP scripts permit calibration prompts to reappear in random evaluation samples; this unit preserves that sampling population.
- Toxicity λ 3.5 is the strongest candidate in the paper-producing Gemma script and is fixed by the paper table's stated rule of maximizing toxicity reduction subject to acceptable PPL.
- Prepare: `python parking/paper_benchmark/paper_benchmark.py --stage prepare --behavior toxicity`.
- Cheap validation: `python parking/paper_benchmark/paper_benchmark.py --stage smoke --behavior toxicity`.
- Two-GPU generation: `python parking/paper_benchmark/paper_benchmark.py --stage generate-pair --behavior toxicity`.
- Two-GPU scoring: `python parking/paper_benchmark/paper_benchmark.py --stage score-pair --behavior toxicity`.
- Summarize and render with `--stage summarize --behavior toxicity --method <method>` and `--stage render --behavior toxicity`.

## References

- A-LQR paper: `ref/2604.19018v1.pdf`.
- Paper-producing toxicity protocol: upstream commit `84b12fa9a9f0af5b6bacbb663debd73d35d0d41c`, `lqr/supertox.py`, `lqr/tox_data_script.py`, `lqr/test_toxicity.py`, and `lqr/testMMLU.py`; perplexity implementation commit `19fd191b79c94d66fc9f8f946ad1df8c4e59de55`, `lqr/ppl_from_file.py`.
- Truthfulness protocol: `ref/lqr-activation-steering/steer/tqa_eval.py` and the sources listed in `ref/NON_HINFINITY_PROTOCOL.md`.

# benchmark_table.md

## Method

- Read only complete TruthfulQA summaries and place T×I, True, and Info into the matching model-method row.
- Leave uncomputed ID, OOD, and capability cells as `TBD`.

## Variables

- Data/input: `cache/results/truthfulness/<model>/<method>.json`.
- Sessions/groups: model-method rows.
- Labels/targets: manuscript truthfulness columns.
- Signals/features/measures: repetition mean ± repetition-level SE.
- Parameters/thresholds: only complete, identity-matched summaries are included.
- Outputs: `plots/benchmark_table.md`.

## Statistics

- Tests/models: descriptive mean ± SE across five repetitions; no inferential test.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: incomplete or missing results remain `TBD`.
- What the statistic means: expected metric value and stochastic repetition variability.
- Why this statistic is appropriate here: each repetition evaluates the full question set.

## Legends

- X axis: table columns are TruthfulQA ID, future shifts, True, Info, and MMLU.
- Y axis: table rows are model-method pairs.
- Color/value: none.
- Grouping: methods within models.
- Ordering/sorting: fixed model and method order.
- Lines/markers/labels: arrows show preferred direction; `TBD` marks unrun cells.
- Panels: none.

## Interpretation

- Compare Original and A-LQR within the same model block.

## Notes

- No values are copied from the invalid 50-question runs.

## References

- `paper_benchmark.py` in this unit.
- `/home/dev/controller/paper/benchmark_table2.tex` for manuscript column organization.

# toxicity_benchmark_table.md

## Method

- Read only complete RTP summaries and place toxicity, Dist-2, MMLU, and PPL into the matching model-method row.
- Leave every unrun method or model as `TBD`.

## Variables

- Data/input: `cache/results/toxicity/<model>/<method>.json`.
- Sessions/groups: model-method rows.
- Labels/targets: toxic versus neutral continuations and correct MMLU answers.
- Signals/features/measures: toxicity percentage, corpus-level distinct bigrams, MMLU accuracy, and prompt-inclusive perplexity.
- Parameters/thresholds: only complete, identity-matched summaries are included.
- Outputs: `plots/toxicity_benchmark_table.md`.

## Statistics

- Tests/models: mean ± SE across five RTP repetitions; MMLU accuracy ± prompt-level Bernoulli SE.
- Null hypothesis: none.
- Alternative hypothesis: none.
- Thresholds/decision rule: incomplete or missing results remain `TBD`.
- What the statistic means: lower toxicity and PPL are better; higher Dist-2 and MMLU are better.
- Why this statistic is appropriate here: it matches the summary structure of the source toxicity table while keeping the MMLU set identical across methods.

## Legends

- X axis: table columns are CLS toxicity, Dist-2, MMLU, and PPL.
- Y axis: table rows are model-method pairs.
- Color/value: none.
- Grouping: methods within models.
- Ordering/sorting: fixed model and method order.
- Lines/markers/labels: arrows show preferred direction; `TBD` marks unrun cells.
- Panels: none.

## Interpretation

- Compare toxicity reduction against diversity, capability, and fluency in the same model block.

## Notes

- Dist-1 and Dist-3 remain in the result cache even though the compact table shows Dist-2, matching the source table layout.

## References

- `paper_benchmark.py` in this unit.
- A-LQR paper Table 1 and Appendix F.
- `/home/dev/controller/paper/benchmark_table1.tex` for manuscript column organization.
