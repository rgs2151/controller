# HarmBench refusal-type analysis

| Model | Method | Direct refusal (%) | Safe partial compliance (%) | Full compliance (%) | USR (%) ↑ |
|---|---|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | Original | 91.83 ± 1.03 | 0.92 ± 0.32 | 7.25 ± 1.21 | 92.75 ± 1.21 |
| Llama-3.2-1B-Instruct | A-LQR | 90.00 ± 1.07 | 2.17 ± 0.83 | 7.83 ± 1.11 | 92.17 ± 1.11 |
| Llama-3.2-1B-Instruct | H∞ (ours) | 96.25 ± 0.78 | 0.88 ± 0.46 | 2.88 ± 0.68 | 97.12 ± 0.68 |
| Llama-3.2-3B-Instruct | Original | 82.33 ± 1.54 | 3.54 ± 1.03 | 14.12 ± 1.18 | 85.88 ± 1.18 |
| Llama-3.2-3B-Instruct | A-LQR | 81.62 ± 1.79 | 4.58 ± 1.28 | 13.79 ± 1.32 | 86.21 ± 1.32 |
| Llama-3.2-3B-Instruct | H∞ (ours) | 91.79 ± 0.69 | 1.83 ± 0.63 | 6.38 ± 0.49 | 93.62 ± 0.49 |
| Llama-3.1-8B-Instruct | Original | 77.25 ± 3.40 | 5.58 ± 2.03 | 17.17 ± 2.61 | 82.83 ± 2.61 |
| Llama-3.1-8B-Instruct | A-LQR | 80.17 ± 3.35 | 6.33 ± 1.46 | 13.50 ± 3.09 | 86.50 ± 3.09 |
| Llama-3.1-8B-Instruct | H∞ (ours) | 91.33 ± 1.81 | 2.33 ± 0.64 | 6.33 ± 1.36 | 93.67 ± 1.36 |

USR is FalseReject's toxic-prompt Useful Safety Rate: Direct Refusal + Safe Partial Compliance. Each cell is the equal-weight mean of the Direct and Adversarial regimes; thus the five jailbreak templates do not outweigh Direct prompts. Per method, the 1B/3B rows contain 240 Direct and 1,200 Adversarial outputs, while the compact 8B row contains 120 Direct and 600 Adversarial outputs. Values are full-sample percentages ± ten-group behavior-clustered jackknife standard errors. The judge uses the verbatim FalseReject Appendix G three-class rubric with pinned `gpt-4o-mini-2024-07-18` at temperature zero.
