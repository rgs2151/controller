# TruthfulQA: Original versus H∞

This analysis-local table is intentionally separate from the paper-facing benchmark tables.
It reports the English TruthfulQA evaluation used by the Gamma Star Addition models.

| Model | Method | True (%) ↑ | Informative (%) ↑ | T×I (%) ↑ | Instruction relevance (0–2) ↑ | Fluency (0–2) ↑ |
|---|---|---:|---:|---:|---:|---:|
| Pythia-14M | Original | 75.40 ± 0.88 | 10.89 ± 1.03 | 8.21 ± 0.77 | 0.11 ± 0.01 | 0.54 ± 0.02 |
| Pythia-14M | H∞ (ours) | 66.10 ± 0.72 | 9.79 ± 1.10 | 6.47 ± 0.77 | 0.00 ± 0.00 | 0.02 ± 0.00 |
| Pythia-31M | Original | 89.84 ± 1.60 | 9.30 ± 0.73 | 8.36 ± 0.63 | 0.14 ± 0.01 | 0.59 ± 0.01 |
| Pythia-31M | H∞ (ours) | 68.42 ± 2.23 | 13.10 ± 0.75 | 8.96 ± 0.56 | 0.01 ± 0.00 | 0.00 ± 0.00 |
| DistilGPT-2 | Original | 64.87 ± 0.72 | 41.49 ± 1.08 | 26.92 ± 0.81 | 0.58 ± 0.01 | 0.90 ± 0.01 |
| DistilGPT-2 | H∞ (ours) | 72.95 ± 0.50 | 30.72 ± 0.78 | 22.41 ± 0.53 | 0.57 ± 0.02 | 0.90 ± 0.01 |
| GPT-2 Small | Original | 48.96 ± 1.64 | 56.18 ± 1.79 | 27.51 ± 0.60 | 0.63 ± 0.03 | 0.94 ± 0.01 |
| GPT-2 Small | H∞ (ours) | 51.04 ± 1.53 | 53.73 ± 2.79 | 27.43 ± 0.74 | 0.57 ± 0.01 | 0.91 ± 0.01 |
| SmolLM2-135M | Original | 47.74 ± 1.17 | 76.25 ± 1.11 | 36.40 ± 0.60 | 0.75 ± 0.02 | 0.94 ± 0.02 |
| SmolLM2-135M | H∞ (ours) | 75.89 ± 1.65 | 64.50 ± 2.20 | 48.95 ± 1.02 | 0.75 ± 0.01 | 0.97 ± 0.01 |
| Pythia-160M | Original | 63.65 ± 0.50 | 41.13 ± 1.45 | 26.18 ± 0.87 | 0.47 ± 0.01 | 0.79 ± 0.01 |
| Pythia-160M | H∞ (ours) | 61.08 ± 1.85 | 34.64 ± 1.53 | 21.16 ± 0.73 | 0.38 ± 0.00 | 0.75 ± 0.01 |
| GPT-2 Medium | Original | 32.68 ± 1.67 | 85.92 ± 0.77 | 28.08 ± 1.26 | 0.74 ± 0.02 | 0.94 ± 0.02 |
| GPT-2 Medium | H∞ (ours) | 32.80 ± 1.25 | 82.86 ± 1.29 | 27.18 ± 0.85 | 0.64 ± 0.01 | 0.81 ± 0.01 |
| Qwen-2.5-0.5B | Original | 50.92 ± 1.73 | 92.04 ± 0.93 | 46.87 ± 2.03 | 0.92 ± 0.03 | 1.02 ± 0.01 |
| Qwen-2.5-0.5B | H∞ (ours) | 64.26 ± 2.96 | 89.23 ± 1.15 | 57.34 ± 2.12 | 0.93 ± 0.03 | 1.01 ± 0.01 |
| GPT-2 Large | Original | 29.50 ± 0.73 | 89.35 ± 1.32 | 26.36 ± 0.89 | 0.80 ± 0.02 | 1.02 ± 0.01 |
| GPT-2 Large | H∞ (ours) | 58.51 ± 2.55 | 64.63 ± 0.54 | 37.81 ± 1.52 | 0.65 ± 0.02 | 0.81 ± 0.01 |
| GPT-2 XL | Original | 28.89 ± 1.56 | 94.25 ± 0.42 | 27.22 ± 1.35 | 0.82 ± 0.01 | 1.08 ± 0.01 |
| GPT-2 XL | H∞ (ours) | 46.51 ± 1.17 | 75.64 ± 1.74 | 35.18 ± 0.43 | 0.78 ± 0.03 | 0.76 ± 0.03 |
| Gemma-2-2B | Original | 49.79 ± 0.22 | 95.35 ± 0.22 | 47.48 ± 0.24 | 1.35 ± 0.02 | 1.23 ± 0.01 |
| Gemma-2-2B | H∞ (ours) | 70.11 ± 0.70 | 79.34 ± 0.50 | 55.62 ± 0.66 | 1.14 ± 0.01 | 1.07 ± 0.01 |
| Llama-3-8B | Original | 49.28 ± 0.98 | 96.99 ± 0.29 | 47.79 ± 0.96 | 1.33 ± 0.03 | 1.30 ± 0.01 |
| Llama-3-8B | H∞ (ours) | 76.50 ± 1.11 | 80.29 ± 0.59 | 61.42 ± 1.15 | 0.99 ± 0.02 | 1.00 ± 0.01 |
| Qwen-2.5-14B | Original | 55.89 ± 0.33 | 96.47 ± 0.15 | 53.92 ± 0.33 | 1.46 ± 0.01 | 1.24 ± 0.01 |
| Qwen-2.5-14B | H∞ (ours) | 85.31 ± 0.33 | 89.40 ± 0.35 | 76.27 ± 0.42 | 1.34 ± 0.01 | 1.15 ± 0.00 |
| OLMo-2-32B | Original | 56.92 ± 2.36 | 98.78 ± 0.34 | 56.22 ± 2.15 | 1.59 ± 0.03 | 1.43 ± 0.02 |
| OLMo-2-32B | H∞ (ours) | 92.66 ± 0.73 | 93.15 ± 0.80 | 86.31 ± 0.42 | 1.38 ± 0.03 | 1.12 ± 0.01 |

## Reading the table

- T×I is the product of the aggregate True and Informative percentages divided by 100; it is not a per-response intersection rate.
- Values are full-sample means ± their stored standard errors. The nine new one-pass runs use five-group delete-one-group question-jackknife uncertainty.
- Instruction relevance and fluency are AXBench judge scores on the 0–2 scale.
- Exact source paths, hashes, run sizes, timestamps, calibration IDs, and uncertainty methods are recorded in `truthfulness_result_sources.json`.
- Nothing in `figs/bench_table/` is read, regenerated, or modified by this unit.
