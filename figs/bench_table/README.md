# Benchmark tables

This unit keeps one self-contained report per evaluated dataset. Each report starts with its result table and then contains only Method, Measures, and Hyperparameters.

## Reports

- Truthfulness: [TruthfulQA](truthfulness/truthfulqa.md) and [Spanish TruthfulQA](truthfulness/truthfulqa_spanish.md).
- Toxicity: [RealToxicityPrompts](toxicity/realtoxicityprompts.md) and [Jigsaw transfer](toxicity/jigsaw.md).

## Required format

The canonical reporting contract is [structure.md](structure.md). Every report must contain:

1. Dataset title and results table.
2. Method and a readable task example.
3. Measures table defining every result column.
4. Hyperparameter table covering every populated model × method row.

Markdown and TeX results are rendered from the same tracked summaries. Regenerate them with:

```bash
python figs/bench_table/bench_table.py
```
