# Benchmark dataset report structure

Create one Markdown report, one synchronized TeX table, and one standalone PDF for every evaluated dataset:

```text
figs/bench_table/
  <benchmark>/
    <dataset>.md
    <dataset>.tex
    <dataset>.pdf
```

The PDF is the primary review artifact. It uses grouped model blocks, rotated model labels, a separated Original row, a low-opacity dark-red primary-metric column, and bold best observed values within each model block. The TeX remains a manuscript-compatible table fragment.

Every Markdown report must use exactly this order.

## 1. Dataset title and results

- Start with the dataset name as the H1.
- Put the results table immediately below the title.
- Use one row per model × method and only the measures scored for that dataset.
- Put each measure's unit, bounded range when applicable, and preferred direction in its column heading.
- Report `mean ± standard error` for repeated evaluations.
- Use `TBD` only when the exact model × method × dataset result is absent.

## 2. Method

- Explain the task, prompt format, evaluation size, repetitions, generation settings, and KV-cache setting.
- Identify the pinned dataset, split, and revision.
- Include one readable example of the real model-facing prompt and, when available, a real saved response or reference answer.
- State whether the dataset is the base dataset or an evaluation-only transfer. For transfer, name the base dataset and state what was inherited without refitting.

## 3. Measures

- Use a table with `Column`, `Meaning`, and `Scorer and decision rule`.
- Define every result column, scorer or deterministic formula, score range, threshold or parsing rule, and preferred direction.
- State how repeated values are summarized.
- When a model judge is used, identify its checkpoint or API model and the exact output structure needed to interpret the score.

## 4. Hyperparameters

- Always include a table with `Model`, `Method`, and `Hyperparameter choice`.
- Include one row for every populated model × method result on that page.
- Record every deployed controller and baseline hyperparameter needed to reproduce that row.
- State whether the choice was source-preserved, manually frozen, calibration-selected, or inherited unchanged from a named base dataset.
- Do not invent settings for `TBD` rows; add their hyperparameter rows only when the corresponding evaluation is frozen.

Do not add separate Variables, Statistics, Legends, Interpretation, Notes, or References sections. Put necessary protocol details inside Method, measure definitions inside Measures, and reproducibility settings inside Hyperparameters.
