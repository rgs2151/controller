# Benchmark tables

Generated from completed controlled-decoding summaries with evaluated-model KV cache disabled. The TeX table is generated from the same rows.

## Toxicity benchmark

| Model | Method | RTP CLS Tox. (%) ↓ | RTP Dist 2 ↑ | RTP PPL ↓ | Jigsaw CLS Tox. (%) ↓ | Jigsaw Dist 2 ↑ | Jigsaw PPL ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma-2-2B | Original | 3.96 ± 0.31 | 0.65 ± 0.00 | 10.98 ± 0.07 | 6.10 ± 0.28 | 0.61 ± 0.00 | 19.19 ± 0.16 |
| Gemma-2-2B | ITI | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | S-PID | 0.16 ± 0.06 | 0.68 ± 0.00 | 11.18 ± 0.04 | 0.52 ± 0.12 | 0.68 ± 0.00 | 18.96 ± 0.13 |
| Gemma-2-2B | A-LQR | 0.00 ± 0.00 | 0.63 ± 0.00 | 24.62 ± 0.17 | 0.00 ± 0.00 | 0.60 ± 0.00 | 28.36 ± 0.23 |
| Gemma-2-2B | H∞ (ours) | 0.12 ± 0.02 | 0.68 ± 0.00 | 14.00 ± 0.08 | 0.16 ± 0.07 | 0.67 ± 0.00 | 21.19 ± 0.16 |
| Llama-3-8B | Original | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ITI | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | A-LQR | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Original | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ITI | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | A-LQR | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD |

Toxicity values are mean ± SE across five 1,000-prompt repetitions.

## Truthfulness benchmark

| Model | Method | ID True (%) ↑ | ID Informative (%) ↑ | ID Instruction relevance (0–2) ↑ | ID Fluency (0–2) ↑ | Spanish True (%) ↑ | Spanish Informative (%) ↑ | Spanish Instruction relevance (0–2) ↑ | Spanish Fluency (0–2) ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma-2-2B | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | A-LQR | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | A-LQR | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | A-LQR | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Truthfulness values are mean ± SE across five complete 817-question repetitions. Scorers run independently; TBD means that scorer has not been run.
