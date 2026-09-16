# Benchmark tables

This unit keeps one self-contained report per evaluated dataset. Each report starts with its result table and then contains only Method, Measures, and Hyperparameters. The sibling PDF is the primary human-review surface; Markdown retains the detailed protocol and TeX remains available for manuscript inclusion.

## Reports

- Truthfulness: [TruthfulQA PDF](truthfulness/truthfulqa.pdf) and [Spanish TruthfulQA PDF](truthfulness/truthfulqa_spanish.pdf).
- Toxicity: [RealToxicityPrompts PDF](toxicity/realtoxicityprompts.pdf) and [Jigsaw transfer PDF](toxicity/jigsaw.pdf).
- MGSM multilingual transfer: [summary PDF](mgsm/mgsm_overall.pdf) and [full per-language PDF](mgsm/mgsm_full.pdf).

## Required format

The canonical reporting contract is [structure.md](structure.md). Every report must contain:

1. Dataset title and results table.
2. Method and a readable task example.
3. Measures table defining every result column.
4. Hyperparameter table covering every populated model × method row.

Markdown, TeX, and standalone PDF results are rendered from the same tracked summaries. Regenerate all three together with:

```bash
python figs/bench_table/bench_table.py
```
