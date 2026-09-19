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
| Gemma-3-4B-Instruct | Chinese | S-PID | 0.0 | 0.00 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Chinese | A-LQR | 0.0 | 0.02 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Chinese | H∞ (ours) | 0.0 | 0.04 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | French | S-PID | 0.0 | 0.00 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | French | A-LQR | 0.0 | 1.02 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | French | H∞ (ours) | 0.0 | 1.10 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Japanese | S-PID | 0.0 | 0.00 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Japanese | A-LQR | 0.0 | 0.00 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Japanese | H∞ (ours) | 0.0 | 0.12 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Swahili | S-PID | 0.0 | 0.00 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Swahili | A-LQR | 0.0 | 0.00 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Swahili | H∞ (ours) | 0.0 | 0.20 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Telugu | S-PID | 0.0 | 0.00 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Telugu | A-LQR | 0.0 | 0.00 | 0.00 | 0.00 |
| Gemma-3-4B-Instruct | Telugu | H∞ (ours) | 0.0 | 0.00 | 0.00 | 0.00 |

## Method

- Task: solve matched MGSM arithmetic problems in Chinese, French, Japanese, Swahili, and Telugu while steering every response toward Spanish. English and Spanish are excluded from evaluation.
- Direction: all 250 matched English–Spanish MGSM pairs define the Spanish steering direction. A-LQR and H∞ share the same 50-Jacobian dynamics estimate. H∞ additionally fits its disturbance geometry and robust controller without changing the shared dynamics matrix.
- Prompting: each language uses its native eight-shot worked-example prompt. Generation is deterministic, limited to 256 new tokens, and runs with evaluated-model KV cache disabled.
- Models: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c` with thinking mode disabled, and `google/gemma-3-4b-it` at revision `093f9f388b31de276ce2de164bdc2081324b9767`.
- Evaluation size: Qwen uses all 250 problems per language for Original, S-PID, A-LQR, and H∞; Gemma uses the frozen 100-problem subset per language for S-PID, A-LQR, and H∞. The summary macro-averages the five language means independently within each model.

## Measures

| Column | Meaning | Scorer and decision rule |
|---|---|---|
| Accuracy (%) ↑ | Percentage of problems with the correct final numeric answer. | Deterministic language-independent parser; commas and a trailing `.0` are normalized, and missing or unparseable answers are incorrect. |
| Spanish relevance (0–2) ↑ | Whether the generated response is in Spanish. | AXBench deterministic Spanish rule evaluator: 0 = rule not satisfied and 2 = rule satisfied. |
| Instruction relevance (0–2) ↑ | Whether the response addresses and attempts the arithmetic task. | AXBench instruction-relevance rubric through `gpt-4o-mini-2024-07-18`; integer score 0, 1, or 2. |
| Fluency (0–2) ↑ | Language quality of the generated response. | AXBench fluency rubric through `gpt-4o-mini-2024-07-18`; integer score 0, 1, or 2. |

These are descriptive means on one fixed evaluation set per language, not repeated trials; therefore the table does not report standard errors. Qwen uses 250 problems per language and Gemma uses the frozen 100-problem subset; every method within a model uses identical problem identities.

## Hyperparameters

| Model | Method | Hyperparameter choice |
|---|---|---|
| Qwen3-4B | Original | No intervention |
| Qwen3-4B | S-PID | λ = 1.5; Kp = 0.5; Ki = 0.5; Kd = 0.01 |
| Qwen3-4B | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Qwen3-4B | H∞ | λ = 1.5; Q/R = 0.01; Qf/R = 0.316227766; R = 1; γ★ = 11.0736; selected on 50 disjoint GSM8K training prompts |
| Gemma-3-4B-Instruct | S-PID | λ = 1.5; Kp = 0.5; Ki = 0.5; Kd = 0.01 |
| Gemma-3-4B-Instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Gemma-3-4B-Instruct | H∞ | R = 1; selected from the frozen 12-point Q/R–Qf/R grid on 50 disjoint GSM8K training prompts |
