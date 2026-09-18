# KV cache investigation

## Method

Compare generation-time KV cache on versus off on the same 100 Spanish TruthfulQA prompts for Original, A-LQR, and H∞. The cache-on condition comes from repetition 0 of the completed benchmark; the cache-off condition is regenerated with every other setting held fixed. Both conditions use the same pinned True and Info judges with judge-side KV caching enabled.

## Variables

- Independent variable: generation-time KV-cache state.
- Groups: Original, A-LQR, and H∞.
- Outcomes: T×I, True percentage, Info percentage, exact answer matches, and matched judge-label flips.

## Statistics

Values are percentages from one matched 100-prompt run. There are no repetitions, standard errors, confidence intervals, or inferential tests.

| Method | KV cache | T×I ↑ | True (%) ↑ | Info (%) ↑ |
|---|---|---:|---:|---:|
| Original | on | 46.17 | 57.00 | 81.00 |
| Original | off | 44.28 | 54.00 | 82.00 |
| A-LQR | on | 57.42 | 87.00 | 66.00 |
| A-LQR | off | 52.56 | 72.00 | 73.00 |
| H∞ | on | 41.36 | 88.00 | 47.00 |
| H∞ | off | 48.64 | 76.00 | 64.00 |

Matched answer-level diagnostics:

| Method | Exact answers (of 100) | True-label flips | Info-label flips |
|---|---:|---:|---:|
| Original | 96 | 3 | 1 |
| A-LQR | 0 | 21 | 25 |
| H∞ | 0 | 24 | 27 |

## Legends

The output is tabular: rows group steering method and KV-cache state; higher T×I, True, and Info are better.

## Interpretation

Original is nearly invariant: 96/100 generated answers are exact matches. A-LQR and H∞ each have 0/100 exact matches, accompanied by substantial judge-label movement. In this hooked feedback pipeline, disabling KV cache therefore changes controller execution and is not merely a speed toggle.

## Notes

This diagnostic does not choose which cache state is mathematically correct. It establishes that the current controller hooks are cache-sensitive. It does not alter benchmark caches, benchmark results, or the benchmark table.

## References

- `benchmarks/truthfulness/`
- `parking/truthfulqa_spanish/`
