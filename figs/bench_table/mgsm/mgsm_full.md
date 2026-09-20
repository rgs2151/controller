# MGSM multilingual transfer — full results

| Model | Language | Method | Accuracy (%) ↑ | Spanish relevance (0–2) ↑ | Instruction relevance (0–2) ↑ | Fluency (0–2) ↑ |
|---|---|---|---:|---:|---:|---:|
| Qwen3-4B | Chinese | Original | 71.0 | 0.00 | 1.80 | 1.71 |
| Qwen3-4B | Chinese | S-PID | 0.0 | 2.00 | 0.00 | 0.00 |
| Qwen3-4B | Chinese | A-LQR | 20.0 | 1.98 | 1.53 | 1.15 |
| Qwen3-4B | Chinese | H∞ (ours) | 51.0 | 2.00 | 1.87 | 1.52 |
| Qwen3-4B | French | Original | 60.0 | 0.00 | 1.93 | 1.70 |
| Qwen3-4B | French | S-PID | 0.0 | 1.92 | 0.00 | 0.00 |
| Qwen3-4B | French | A-LQR | 29.0 | 1.94 | 1.62 | 1.11 |
| Qwen3-4B | French | H∞ (ours) | 48.0 | 1.96 | 1.96 | 1.42 |
| Qwen3-4B | Japanese | Original | 56.0 | 0.00 | 1.80 | 1.64 |
| Qwen3-4B | Japanese | S-PID | 0.0 | 2.00 | 0.00 | 0.00 |
| Qwen3-4B | Japanese | A-LQR | 32.0 | 1.84 | 1.63 | 1.01 |
| Qwen3-4B | Japanese | H∞ (ours) | 40.0 | 1.98 | 1.71 | 1.20 |
| Qwen3-4B | Swahili | Original | 15.0 | 0.00 | 1.43 | 1.23 |
| Qwen3-4B | Swahili | S-PID | 0.0 | 1.96 | 0.00 | 0.00 |
| Qwen3-4B | Swahili | A-LQR | 12.0 | 1.40 | 1.32 | 0.97 |
| Qwen3-4B | Swahili | H∞ (ours) | 14.0 | 0.92 | 1.44 | 1.06 |
| Qwen3-4B | Telugu | Original | 6.0 | 0.00 | 1.83 | 1.05 |
| Qwen3-4B | Telugu | S-PID | 0.0 | 2.00 | 0.00 | 0.00 |
| Qwen3-4B | Telugu | A-LQR | 19.0 | 1.88 | 1.46 | 1.18 |
| Qwen3-4B | Telugu | H∞ (ours) | 8.0 | 0.34 | 1.33 | 1.05 |
| Llama-3.2-3B | Chinese | Original | 43.0 | 0.00 | 1.79 | 1.59 |
| Llama-3.2-3B | Chinese | S-PID | 9.0 | 1.96 | 1.23 | 0.87 |
| Llama-3.2-3B | Chinese | A-LQR | 26.0 | 1.80 | 1.49 | 1.21 |
| Llama-3.2-3B | Chinese | H∞ (ours) | 34.0 | 0.92 | 1.50 | 1.17 |
| Llama-3.2-3B | French | Original | 49.0 | 0.00 | 1.83 | 1.71 |
| Llama-3.2-3B | French | S-PID | 14.0 | 1.98 | 1.21 | 0.67 |
| Llama-3.2-3B | French | A-LQR | 43.0 | 1.24 | 1.76 | 1.09 |
| Llama-3.2-3B | French | H∞ (ours) | 42.0 | 0.80 | 1.77 | 1.10 |
| Llama-3.2-3B | Japanese | Original | 27.0 | 0.00 | 1.56 | 1.34 |
| Llama-3.2-3B | Japanese | S-PID | 10.0 | 1.96 | 0.67 | 0.63 |
| Llama-3.2-3B | Japanese | A-LQR | 26.0 | 1.94 | 1.68 | 1.31 |
| Llama-3.2-3B | Japanese | H∞ (ours) | 24.0 | 0.96 | 1.56 | 1.28 |
| Llama-3.2-3B | Swahili | Original | 40.0 | 0.00 | 1.79 | 1.49 |
| Llama-3.2-3B | Swahili | S-PID | 2.0 | 1.90 | 0.47 | 0.38 |
| Llama-3.2-3B | Swahili | A-LQR | 22.0 | 0.68 | 1.47 | 0.98 |
| Llama-3.2-3B | Swahili | H∞ (ours) | 19.0 | 0.52 | 1.38 | 0.95 |
| Llama-3.2-3B | Telugu | Original | 6.0 | 0.00 | 1.43 | 1.07 |
| Llama-3.2-3B | Telugu | S-PID | 6.0 | 1.90 | 0.79 | 0.46 |
| Llama-3.2-3B | Telugu | A-LQR | 21.0 | 1.86 | 1.44 | 1.25 |
| Llama-3.2-3B | Telugu | H∞ (ours) | 10.0 | 0.96 | 1.08 | 0.92 |

## Method

- Task: solve matched MGSM arithmetic problems in Chinese, French, Japanese, Swahili, and Telugu while steering every response toward Spanish. English and Spanish are excluded from evaluation.
- Direction: all 250 matched English–Spanish MGSM pairs define the Spanish steering direction. A-LQR and H∞ share the same 50-Jacobian dynamics estimate. H∞ additionally fits its disturbance geometry and robust controller without changing the shared dynamics matrix.
- Prompting: each language uses its native eight-shot worked-example prompt. Generation is deterministic, limited to 256 new tokens, and runs with evaluated-model KV cache disabled.
- Models: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c` with thinking mode disabled, and `meta-llama/Llama-3.2-3B-Instruct` at revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
- Evaluation size: Qwen uses all 250 problems per language and Llama uses the frozen 100-problem subset per language. Both models report Original, S-PID, A-LQR, and H∞ on identical problem identities within each model. The summary macro-averages the five language means independently within each model.

## Measures

| Column | Meaning | Scorer and decision rule |
|---|---|---|
| Accuracy (%) ↑ | Percentage of problems with the correct final numeric answer. | Deterministic language-independent parser; commas and a trailing `.0` are normalized, and missing or unparseable answers are incorrect. |
| Spanish relevance (0–2) ↑ | Whether the generated response is in Spanish. | AXBench deterministic Spanish rule evaluator: 0 = rule not satisfied and 2 = rule satisfied. |
| Instruction relevance (0–2) ↑ | Whether the response addresses and attempts the arithmetic task. | AXBench instruction-relevance rubric through `gpt-4o-mini-2024-07-18`; integer score 0, 1, or 2. |
| Fluency (0–2) ↑ | Language quality of the generated response. | AXBench fluency rubric through `gpt-4o-mini-2024-07-18`; integer score 0, 1, or 2. |

These are descriptive means on one fixed evaluation set per language, not repeated trials; therefore the table does not report standard errors. Qwen uses 250 problems per language and Llama uses the frozen 100-problem subset; every method within a model uses identical problem identities.

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
| Llama-3.2-3B-Instruct | H∞ | λ = 1.5 fixed; Q/R = 0.01; Qf/R = 0.01; R = 1; γ★ = 0.284523; costs selected from the frozen 12-point grid on 50 disjoint GSM8K training prompts using the equal-weight additive combination of exact-answer accuracy and normalized AXBench Overall; no λ sweep |
