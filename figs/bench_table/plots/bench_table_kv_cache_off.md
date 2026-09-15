# Benchmark tables — evaluated-model KV cache off

Generated from completed summaries in `benchmarks/*/results/kv_cache_off/`. The TeX table is generated from the same rows.

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

Toxicity, Dist-2, and PPL are mean ± SE across five complete 1,000-prompt repetitions for both RTP and Jigsaw.

## Truthfulness benchmark

| Model | Method | TruthfulQA–ID T×I ↑ | ID True (%) ↑ | ID Info (%) ↑ | ID Instruction relevance (0–2) ↑ | ID Fluency (0–2) ↑ | Spanish T×I ↑ | Spanish True (%) ↑ | Spanish Info (%) ↑ | Spanish Instruction relevance (0–2) ↑ | Spanish Fluency (0–2) ↑ | MMLU (%) ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma-2-2B | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | A-LQR | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | A-LQR | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | A-LQR | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Truthfulness, AXBench instruction relevance (0–2), and AXBench fluency (0–2) are mean ± SE across five complete 817-question repetitions. TBD cells have not been run.
