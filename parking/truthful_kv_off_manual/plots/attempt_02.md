# Truthful KV-off manual — attempt 02

## Method

Generate answers for the same 50 TruthfulQA questions in the original English form and the frozen Spanish translation, using Gemma-2-2B with generation-time KV caching disabled. Compare Original, published-setting A-LQR, and H∞ synthesized manually at Q=0.316227766017, R=3, and Qf=0.1, without a hyperparameter sweep.

## Variables

- Distribution: ID English or Spanish question with an English-answer instruction.
- Method: Original, A-LQR, or manually configured H∞.
- Outcomes: aggregate T×I, True percentage, and Info percentage.

## Statistics

This is one descriptive run of 50 matched questions per condition. T×I is aggregate True percentage multiplied by aggregate Info percentage and divided by 100. No repetition, uncertainty interval, or inferential test is used.

| Method | ID T×I ↑ | ID True (%) ↑ | ID Info (%) ↑ | Spanish T×I ↑ | Spanish True (%) ↑ | Spanish Info (%) ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Original | 45.08 | 46.00 | 98.00 | 44.28 | 54.00 | 82.00 |
| A-LQR | 52.64 | 56.00 | 94.00 | 54.40 | 68.00 | 80.00 |
| H∞ manual | 57.12 | 68.00 | 84.00 | 44.64 | 72.00 | 62.00 |

## Legends

The output is a table. Rows are methods; the first three metric columns are ID and the final three are Spanish. Higher values are better.

## Interpretation

On ID, manual H∞ has the highest T×I (57.12), ahead of A-LQR (52.64) and Original (45.08). On Spanish, A-LQR has the highest T×I (54.40), ahead of manual H∞ (44.64) and Original (44.28). Manual H∞ raises True more than A-LQR in both distributions, but its lower Info score erases that advantage on Spanish.

## Notes

H∞ uses λ=3, Q=0.316227766017, R=3, Qf=0.1, Q/R=0.105409255339, Qf/R=0.0333333333333, and γ*=25.2155423164. Judge-side KV caching remains enabled so only the evaluated model's decoding policy changes.

## References

- `benchmarks/truthfulness/`
- `parking/truthfulqa_spanish/`
- `parking/kv_cache_investigation/`
