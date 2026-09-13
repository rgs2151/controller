# Benchmark tables

Generated from completed summaries in `parking/bench_evaluations/cache/results/`. The TeX table is generated from the same rows.

## Toxicity benchmark

| Model | Method | CLS Tox. (%) ↓ | Dist 2 ↑ | MMLU (%) ↑ | PPL ↓ |
|---|---|---:|---:|---:|---:|
| Gemma-2-2B | Original | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ITI | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ActAdd | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Mean-AcT | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Linear-AcT | TBD | TBD | TBD | TBD |
| Gemma-2-2B | PID-AcT | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ODESteer | TBD | TBD | TBD | TBD |
| Gemma-2-2B | S-PID | TBD | TBD | TBD | TBD |
| Gemma-2-2B | A-LQR | TBD | TBD | TBD | TBD |
| Gemma-2-2B | H∞ (ours) | TBD | TBD | TBD | TBD |
| Llama-3-8B | Original | TBD | TBD | TBD | TBD |
| Llama-3-8B | ITI | TBD | TBD | TBD | TBD |
| Llama-3-8B | ActAdd | TBD | TBD | TBD | TBD |
| Llama-3-8B | Mean-AcT | TBD | TBD | TBD | TBD |
| Llama-3-8B | Linear-AcT | TBD | TBD | TBD | TBD |
| Llama-3-8B | PID-AcT | TBD | TBD | TBD | TBD |
| Llama-3-8B | ODESteer | TBD | TBD | TBD | TBD |
| Llama-3-8B | S-PID | TBD | TBD | TBD | TBD |
| Llama-3-8B | A-LQR | TBD | TBD | TBD | TBD |
| Llama-3-8B | H∞ (ours) | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Original | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ITI | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ActAdd | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Mean-AcT | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Linear-AcT | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | PID-AcT | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ODESteer | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | S-PID | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | A-LQR | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | H∞ (ours) | TBD | TBD | TBD | TBD |

Toxicity, Dist-2, and PPL are mean ± SE across five complete 1,000-prompt RTP repetitions. MMLU is accuracy ± prompt-level SE on one shared 1,000-question 5-shot set.

## Truthfulness benchmark

| Model | Method | TruthfulQA–ID T×I ↑ | Spanish | Adversarial | Long | True (%) ↑ | Info (%) ↑ | MMLU (%) ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma-2-2B | Original | 47.59 ± 0.38 | TBD | TBD | TBD | 50.04 ± 0.24 | 95.10 ± 0.33 | TBD |
| Gemma-2-2B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | A-LQR | 66.89 ± 0.40 | TBD | TBD | TBD | 75.69 ± 0.45 | 88.37 ± 0.23 | TBD |
| Gemma-2-2B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Original | 46.26 ± 1.20 | TBD | TBD | TBD | 47.54 ± 1.27 | 97.31 ± 0.21 | TBD |
| Llama-3-8B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | A-LQR | 64.61 ± 0.40 | TBD | TBD | TBD | 66.22 ± 0.26 | 97.58 ± 0.27 | TBD |
| Llama-3-8B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Original | 49.35 ± 0.55 | TBD | TBD | TBD | 51.26 ± 0.54 | 96.28 ± 0.22 | TBD |
| Qwen-2.5-14B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | A-LQR | 77.38 ± 0.44 | TBD | TBD | TBD | 80.27 ± 0.42 | 96.40 ± 0.22 | TBD |
| Qwen-2.5-14B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Truthfulness values are mean ± SE across five complete 817-question repetitions. TBD cells have not been run.
