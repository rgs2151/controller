# Benchmark tables — evaluated-model KV cache on

Generated from completed summaries in `benchmarks/*/results/kv_cache_on/`. The TeX table is generated from the same rows.

## Toxicity benchmark

| Model | Method | RTP CLS Tox. (%) ↓ | RTP Dist 2 ↑ | RTP PPL ↓ | Jigsaw CLS Tox. (%) ↓ | Jigsaw Dist 2 ↑ | Jigsaw PPL ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma-2-2B | Original | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ITI | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | A-LQR | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD |
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
| Gemma-2-2B | Original | 47.59 ± 0.38 | 50.04 ± 0.24 | 95.10 ± 0.33 | 1.35 ± 0.01 | 1.35 ± 0.01 | 45.27 ± 0.64 | 51.87 ± 0.73 | 87.27 ± 0.43 | 1.33 ± 0.01 | 1.27 ± 0.01 | TBD |
| Gemma-2-2B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gemma-2-2B | A-LQR | 66.89 ± 0.40 | 75.69 ± 0.45 | 88.37 ± 0.23 | 1.25 ± 0.01 | 1.18 ± 0.01 | 57.87 ± 0.55 | 85.07 ± 0.46 | 68.03 ± 0.54 | 1.06 ± 0.01 | 1.09 ± 0.00 | TBD |
| Gemma-2-2B | H∞ (ours) | 59.15 ± 0.29 | 80.07 ± 0.44 | 73.88 ± 0.56 | 1.12 ± 0.02 | 1.06 ± 0.00 | 47.49 ± 0.65 | 85.85 ± 0.25 | 55.32 ± 0.87 | 1.04 ± 0.01 | 1.06 ± 0.01 | TBD |
| Llama-3-8B | Original | 46.26 ± 1.20 | 47.54 ± 1.27 | 97.31 ± 0.21 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | A-LQR | 64.61 ± 0.40 | 66.22 ± 0.26 | 97.58 ± 0.27 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Llama-3-8B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Original | 49.35 ± 0.55 | 51.26 ± 0.54 | 96.28 ± 0.22 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ITI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ActAdd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Mean-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | Linear-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | PID-AcT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | ODESteer | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | S-PID | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | A-LQR | 77.38 ± 0.44 | 80.27 ± 0.42 | 96.40 ± 0.22 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen-2.5-14B | H∞ (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Truthfulness, AXBench instruction relevance (0–2), and AXBench fluency (0–2) are mean ± SE across five complete 817-question repetitions. TBD cells have not been run.
