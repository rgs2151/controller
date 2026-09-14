# attempt_01

## Method

- Select the first 50 matched question IDs from repetition 0 of the frozen English and Spanish TruthfulQA datasets.
- Generate one answer per question with Gemma-2-2B under Original, published-setting A-LQR, and manually configured H∞, with evaluated-model KV caching disabled.
- Reuse the frozen TruthfulQA direction, setpoints, dynamics, reduced coordinates, and disturbance geometry.
- Synthesize H∞ at λ=3, Q=0.1, R=1, and Qf=0.316227766 without a sweep.
- Score each answer with the pinned TruthfulQA True and Info judges and write `plots/attempt_01.md`.

## Variables

- Data/input: the frozen TruthfulQA benchmark dataset and `parking/truthfulqa_spanish/data/truthfulqa_spanish.json`.
- Sessions/groups: ID English and Spanish translation; Original, A-LQR, and manual H∞.
- Labels/targets: exact case-insensitive `yes` from the True or Helpful judge scores 1; every other judge output scores 0.
- Signals/features/measures: generated English answer, True percentage, Info percentage, and aggregate T×I.
- Parameters/thresholds: 50 matched prompts per distribution; one run; seed 42; generation batch size 8; generation KV cache off; H∞ Q/R=0.1 and Qf/R=0.316227766.
- Outputs: ignored controller, generation, score, timing, and provenance caches under `cache/attempts/attempt_01/`; tracked table `plots/attempt_01.md`.

## Statistics

- Tests/models: descriptive percentages only; no inferential test.
- Null hypothesis: not tested.
- Alternative hypothesis: not tested.
- Thresholds/decision rule: compare methods within each distribution and compare each method between ID and Spanish.
- What the statistic means: True and Info are judge acceptance rates; T×I is their aggregate product divided by 100.
- Why this statistic is appropriate here: it matches the TruthfulQA benchmark metric while exposing both components on paired questions.

## Legends

- X axis: none; the output is a table.
- Y axis: none.
- Color/value: none.
- Grouping: method rows and distribution-specific metric columns.
- Ordering/sorting: Original, A-LQR, H∞; ID before Spanish; T×I, True, Info within each distribution.
- Lines/markers/labels: H∞ is labeled manual to distinguish it from the calibrated benchmark controller.
- Panels: none.

## Interpretation

- On ID, manual H∞ has the highest T×I (60.68), ahead of A-LQR (52.64) and Original (45.08).
- On Spanish, A-LQR has the highest T×I (54.40), ahead of manual H∞ (49.60) and Original (44.28).
- H∞ increases True but loses Info under Spanish transfer.

## Notes

- H∞ uses γ*=40.3948009014.
- Judge-side KV caching remains enabled because the judge has no intervention hooks.
- This attempt does not modify benchmark-owned artifacts or tables.

## References

- `benchmarks/truthfulness/`
- `parking/truthfulqa_spanish/`
- `parking/kv_cache_investigation/`

# attempt_02

## Method

- Use the same 50 matched ID and Spanish questions, generation settings, seed, model, judges, and A-LQR controller as Attempt 1.
- Rerun Original, A-LQR, and H∞ from scratch with evaluated-model KV caching disabled.
- Reuse the same frozen H∞ state, target, dynamics, and disturbance artifacts, but resynthesize its gains at λ=3, Q=0.316227766, R=3, and Qf=0.1 without a sweep.
- Score each answer with the pinned TruthfulQA True and Info judges and write `plots/attempt_02.md`.

## Variables

- Data/input: the same paired 50-row subset used by Attempt 1.
- Sessions/groups: ID English and Spanish translation; Original, A-LQR, and manual H∞.
- Labels/targets: exact case-insensitive `yes` from the True or Helpful judge scores 1; every other judge output scores 0.
- Signals/features/measures: generated English answer, True percentage, Info percentage, and aggregate T×I.
- Parameters/thresholds: one run; seed 42; generation batch size 8; generation KV cache off; H∞ Q/R=0.105409255 and Qf/R=0.033333333.
- Outputs: ignored controller, generation, score, timing, and provenance caches under `cache/attempts/attempt_02/`; tracked table `plots/attempt_02.md`.

## Statistics

- Tests/models: descriptive percentages only; no inferential test.
- Null hypothesis: not tested.
- Alternative hypothesis: not tested.
- Thresholds/decision rule: compare methods within each distribution and compare Attempt 2 H∞ with Attempt 1 H∞.
- What the statistic means: True and Info are judge acceptance rates; T×I is their aggregate product divided by 100.
- Why this statistic is appropriate here: it keeps the evaluation fixed while changing only the H∞ cost configuration.

## Legends

- X axis: none; the output is a table.
- Y axis: none.
- Color/value: none.
- Grouping: method rows and distribution-specific metric columns.
- Ordering/sorting: Original, A-LQR, H∞; ID before Spanish; T×I, True, Info within each distribution.
- Lines/markers/labels: H∞ is labeled manual to distinguish it from the calibrated benchmark controller.
- Panels: none.

## Interpretation

- On ID, manual H∞ has T×I 57.12, compared with 52.64 for A-LQR and 45.08 for Original.
- On Spanish, manual H∞ has T×I 44.64, compared with 54.40 for A-LQR and 44.28 for Original.
- Relative to Attempt 1, Attempt 2 reduces H∞ T×I from 60.68 to 57.12 on ID and from 49.60 to 44.64 on Spanish.

## Notes

- H∞ uses γ*=25.2155423164.
- Judge-side KV caching remains enabled because the judge has no intervention hooks.
- This attempt does not modify benchmark-owned artifacts or tables.

## References

- `plots/attempt_01.md`
- `benchmarks/truthfulness/`
- `parking/truthfulqa_spanish/`
