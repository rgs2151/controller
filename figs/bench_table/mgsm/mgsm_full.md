# MGSM multilingual transfer — full results

| Model | Language | Method | Accuracy (%) ↑ | Spanish relevance (0–2) ↑ | Instruction relevance (0–2) ↑ | Fluency (0–2) ↑ |
|---|---|---|---:|---:|---:|---:|
| Qwen3-4B | Chinese | Original | 71.0 ± 4.1 | 0.00 ± 0.00 | 1.80 ± 0.05 | 1.71 ± 0.05 |
| Qwen3-4B | Chinese | S-PID | 0.0 ± 0.0 | 2.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Qwen3-4B | Chinese | A-LQR | 20.0 ± 2.6 | 1.98 ± 0.02 | 1.53 ± 0.03 | 1.15 ± 0.03 |
| Qwen3-4B | Chinese | H∞ (ours) | 51.0 ± 3.5 | 2.00 ± 0.00 | 1.87 ± 0.05 | 1.52 ± 0.04 |
| Qwen3-4B | French | Original | 60.0 ± 4.5 | 0.00 ± 0.00 | 1.93 ± 0.03 | 1.70 ± 0.04 |
| Qwen3-4B | French | S-PID | 0.0 ± 0.0 | 1.92 ± 0.03 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Qwen3-4B | French | A-LQR | 29.0 ± 5.0 | 1.94 ± 0.04 | 1.62 ± 0.05 | 1.11 ± 0.03 |
| Qwen3-4B | French | H∞ (ours) | 48.0 ± 4.7 | 1.96 ± 0.04 | 1.96 ± 0.02 | 1.42 ± 0.07 |
| Qwen3-4B | Japanese | Original | 56.0 ± 4.8 | 0.00 ± 0.00 | 1.80 ± 0.04 | 1.64 ± 0.04 |
| Qwen3-4B | Japanese | S-PID | 0.0 ± 0.0 | 2.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Qwen3-4B | Japanese | A-LQR | 32.0 ± 2.5 | 1.84 ± 0.06 | 1.63 ± 0.06 | 1.01 ± 0.03 |
| Qwen3-4B | Japanese | H∞ (ours) | 40.0 ± 4.9 | 1.98 ± 0.02 | 1.71 ± 0.07 | 1.20 ± 0.05 |
| Qwen3-4B | Swahili | Original | 15.0 ± 4.3 | 0.00 ± 0.00 | 1.43 ± 0.07 | 1.23 ± 0.05 |
| Qwen3-4B | Swahili | S-PID | 0.0 ± 0.0 | 1.96 ± 0.03 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Qwen3-4B | Swahili | A-LQR | 12.0 ± 3.6 | 1.40 ± 0.09 | 1.32 ± 0.07 | 0.97 ± 0.03 |
| Qwen3-4B | Swahili | H∞ (ours) | 14.0 ± 3.7 | 0.92 ± 0.12 | 1.44 ± 0.07 | 1.06 ± 0.04 |
| Qwen3-4B | Telugu | Original | 6.0 ± 2.7 | 0.00 ± 0.00 | 1.83 ± 0.04 | 1.05 ± 0.02 |
| Qwen3-4B | Telugu | S-PID | 0.0 ± 0.0 | 2.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Qwen3-4B | Telugu | A-LQR | 19.0 ± 2.8 | 1.88 ± 0.04 | 1.46 ± 0.06 | 1.18 ± 0.03 |
| Qwen3-4B | Telugu | H∞ (ours) | 8.0 ± 2.5 | 0.34 ± 0.08 | 1.33 ± 0.04 | 1.05 ± 0.03 |
| Llama-3.2-3B | Chinese | Original | 43.0 ± 3.3 | 0.00 ± 0.00 | 1.79 ± 0.07 | 1.59 ± 0.07 |
| Llama-3.2-3B | Chinese | S-PID | 9.0 ± 2.3 | 1.96 ± 0.03 | 1.23 ± 0.05 | 0.87 ± 0.06 |
| Llama-3.2-3B | Chinese | A-LQR | 26.0 ± 3.7 | 1.80 ± 0.04 | 1.49 ± 0.08 | 1.21 ± 0.08 |
| Llama-3.2-3B | Chinese | H∞ (ours) | 13.0 ± 3.0 | 1.98 ± 0.02 | 1.48 ± 0.05 | 1.18 ± 0.06 |
| Llama-3.2-3B | French | Original | 49.0 ± 4.6 | 0.00 ± 0.00 | 1.83 ± 0.03 | 1.71 ± 0.04 |
| Llama-3.2-3B | French | S-PID | 14.0 ± 3.4 | 1.98 ± 0.02 | 1.21 ± 0.04 | 0.67 ± 0.06 |
| Llama-3.2-3B | French | A-LQR | 43.0 ± 4.2 | 1.24 ± 0.09 | 1.76 ± 0.04 | 1.09 ± 0.03 |
| Llama-3.2-3B | French | H∞ (ours) | 8.0 ± 2.9 | 1.00 ± 0.07 | 0.79 ± 0.05 | 0.69 ± 0.05 |
| Llama-3.2-3B | Japanese | Original | 27.0 ± 3.0 | 0.00 ± 0.00 | 1.56 ± 0.08 | 1.34 ± 0.09 |
| Llama-3.2-3B | Japanese | S-PID | 10.0 ± 2.6 | 1.96 ± 0.04 | 0.67 ± 0.06 | 0.63 ± 0.09 |
| Llama-3.2-3B | Japanese | A-LQR | 26.0 ± 3.7 | 1.94 ± 0.04 | 1.68 ± 0.08 | 1.31 ± 0.07 |
| Llama-3.2-3B | Japanese | H∞ (ours) | 13.0 ± 3.0 | 2.00 ± 0.00 | 1.29 ± 0.05 | 1.08 ± 0.08 |
| Llama-3.2-3B | Swahili | Original | 40.0 ± 3.3 | 0.00 ± 0.00 | 1.79 ± 0.02 | 1.49 ± 0.03 |
| Llama-3.2-3B | Swahili | S-PID | 2.0 ± 1.3 | 1.90 ± 0.04 | 0.47 ± 0.10 | 0.38 ± 0.05 |
| Llama-3.2-3B | Swahili | A-LQR | 22.0 ± 4.4 | 0.68 ± 0.10 | 1.47 ± 0.04 | 0.98 ± 0.04 |
| Llama-3.2-3B | Swahili | H∞ (ours) | 18.0 ± 4.2 | 1.96 ± 0.03 | 1.27 ± 0.07 | 0.93 ± 0.06 |
| Llama-3.2-3B | Telugu | Original | 6.0 ± 2.2 | 0.00 ± 0.00 | 1.43 ± 0.04 | 1.07 ± 0.03 |
| Llama-3.2-3B | Telugu | S-PID | 6.0 ± 2.7 | 1.90 ± 0.03 | 0.79 ± 0.08 | 0.46 ± 0.06 |
| Llama-3.2-3B | Telugu | A-LQR | 21.0 ± 6.0 | 1.86 ± 0.05 | 1.44 ± 0.08 | 1.25 ± 0.07 |
| Llama-3.2-3B | Telugu | H∞ (ours) | 3.0 ± 1.5 | 1.36 ± 0.06 | 0.84 ± 0.09 | 0.79 ± 0.07 |

## Method

- Task: solve matched MGSM arithmetic problems in Chinese, French, Japanese, Swahili, and Telugu while steering every response toward Spanish. English and Spanish are excluded from evaluation.
- Direction: all 250 matched English–Spanish MGSM pairs define the Spanish steering direction. A-LQR and H∞ share the same 50-Jacobian dynamics estimate. H∞ additionally fits its disturbance geometry and robust controller without changing the shared dynamics matrix.
- Prompting: each language uses its native eight-shot worked-example prompt. Generation is deterministic, limited to 256 new tokens, and runs with evaluated-model KV cache disabled.
- Models: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c` with thinking mode disabled, and `meta-llama/Llama-3.2-3B-Instruct` at revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
- Evaluation size: both models use the same frozen 100-problem subset per language. Both report Original, S-PID, A-LQR, and H∞ on identical problem identities within each model. The summary macro-averages the five language means independently within each model.

## Measures

| Column | Meaning | Scorer and decision rule |
|---|---|---|
| Accuracy (%) ↑ | Percentage of problems with the correct final numeric answer. | Deterministic language-independent parser; commas and a trailing `.0` are normalized, and missing or unparseable answers are incorrect. |
| Spanish relevance (0–2) ↑ | Whether the generated response is in Spanish. | AXBench deterministic Spanish rule evaluator: 0 = rule not satisfied and 2 = rule satisfied. |
| Instruction relevance (0–2) ↑ | Whether the response addresses and attempts the arithmetic task. | AXBench instruction-relevance rubric through `gpt-4o-mini-2024-07-18`; integer score 0, 1, or 2. |
| Fluency (0–2) ↑ | Language quality of the generated response. | AXBench fluency rubric through `gpt-4o-mini-2024-07-18`; integer score 0, 1, or 2. |

Values are full-sample means ± ten-group delete-one-group jackknife standard errors. Each group contains 10 matched MGSM question identities; deleting a group removes the same questions from all five languages. The uncertainty measures question-sampling variability, not decoding-run or judge variability. Every method within a model uses identical problem identities.

## Hyperparameters

| Model | Method | Hyperparameter choice |
|---|---|---|
| Qwen3-4B | Original | No intervention |
| Qwen3-4B | S-PID | λ = 1.5; Kp = 0.5; Ki = 0.5; Kd = 0.01 |
| Qwen3-4B | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Qwen3-4B | H∞ | λ = 1.5; Q/R = 0.01; Qf/R = 0.316227766; R = 1; γ★ = 11.0736; selected on 50 disjoint GSM8K training prompts |
| Llama-3.2-3B-Instruct | Original | No intervention |
| Llama-3.2-3B-Instruct | S-PID | λ = 1.5; Kp = 0.5; Ki = 0.5; Kd = 0.01 |
| Llama-3.2-3B-Instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Llama-3.2-3B-Instruct | H∞ | λ = 1.5 fixed; Q/R = 10; Qf/R = 0.01; R = 1; γ★ = 4.951513; costs selected from the frozen 12-point grid on 50 disjoint translated GSM8K training prompts balanced across Bengali, German, Russian, and Thai using the equal-weight additive combination of exact-answer accuracy and normalized AXBench Overall; no λ sweep |
