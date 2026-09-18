# kv_cache_investigation

## Method

- Select the first 100 prompts from repetition 0 of the frozen Spanish TruthfulQA set; this repetition is already a seed-42 permutation of all 817 questions.
- Compare Original, A-LQR, and H∞ on exactly the same prompt IDs, order, generation seed, batch boundaries, model checkpoint, controller artifacts, generation settings, and English-answer instruction.
- Read the KV-cache-on generations and pinned True/Helpful judge labels from the completed benchmark.
- Generate one fresh 100-prompt run per method with generation-time KV caching disabled, then score every answer with the same pinned judges and rubrics used by the benchmark.
- Report T×I, True percentage, and Info percentage for cache on and cache off. T×I is the product of aggregate True and Info percentages divided by 100.

## Variables

- Data/input: `parking/truthfulqa_spanish/data/truthfulqa_spanish.json`, repetition 0, rows 0–99.
- Sessions/groups: one 100-prompt run for each of Original, A-LQR, and H∞ under each cache state.
- Labels/targets: exact case-insensitive `yes` from the True or Helpful judge scores 1; every other judge output scores 0.
- Signals/features/measures: generated English answer, True judge label, Info judge label, True percentage, Info percentage, and T×I.
- Parameters/thresholds: Gemma-2-2B revision `c5ebcd4...`; generation seed 42; batch size 8; 50-token maximum; temperature 1; top-p 0.3; repetition penalty 1.2.
- Outputs: ignored generation and score caches under `cache/`; tracked comparison table at `plots/kv_cache_investigation.md`.

## Statistics

- Tests/models: descriptive percentages only; no inferential test.
- Null hypothesis: not tested.
- Alternative hypothesis: not tested.
- Thresholds/decision rule: compare cache-on and cache-off values directly on the same 100 prompts; no uncertainty interval is reported because there is one run.
- What the statistic means: True and Info are the fractions accepted by their respective judges; T×I penalizes a method when either aggregate component is low.
- Why this statistic is appropriate here: holding prompt identities and every non-cache setting fixed isolates whether generation-time KV caching materially changes the benchmark metrics.

## Legends

- X axis: none; the output is a table.
- Y axis: none.
- Color/value: none.
- Grouping: steering method and generation-time KV-cache state.
- Ordering/sorting: Original, A-LQR, H∞; cache on before cache off.
- Lines/markers/labels: none.
- Panels: none.

## Interpretation

- The completed comparison is in `plots/kv_cache_investigation.md`.
- Original is nearly invariant, with 96 of 100 answers exactly matching between cache states.
- A-LQR and H∞ each change all 100 answers, with substantial True and Info label movement. Generation-time KV caching is therefore part of controller execution semantics in the current hooked feedback pipeline, not merely a runtime optimization.

## Notes

- Only model generation changes cache state. Both judge models keep KV caching enabled in both conditions.
- A-LQR and H∞ reuse the frozen ID controllers without fitting Jacobians, setpoints, disturbances, or hyperparameters.
- This investigation does not modify the benchmark table or any benchmark-owned cache.

## References

- `benchmarks/truthfulness/`
- `parking/truthfulqa_spanish/`
