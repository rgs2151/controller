# MGSM multilingual transfer — summary

| Model | Method | Accuracy (%) ↑ | Spanish relevance (0–2) ↑ | Instruction relevance (0–2) ↑ | Fluency (0–2) ↑ |
|---|---|---:|---:|---:|---:|
| Qwen3-4B | Original | 41.6 | 0.00 | 1.76 | 1.47 |
| Qwen3-4B | S-PID | 0.0 | 1.98 | 0.00 | 0.00 |
| Qwen3-4B | A-LQR | 22.4 | 1.81 | 1.51 | 1.08 |
| Qwen3-4B | H∞ (ours) | 32.2 | 1.44 | 1.66 | 1.25 |
| Llama-3.2-3B | Original | 33.0 | 0.00 | 1.68 | 1.44 |
| Llama-3.2-3B | S-PID | 8.2 | 1.94 | 0.87 | 0.60 |
| Llama-3.2-3B | A-LQR | 27.6 | 1.50 | 1.57 | 1.17 |
| Llama-3.2-3B | H∞ (ours) | 27.4 | 0.01 | 1.64 | 1.41 |

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
| Llama-3.2-3B-Instruct | H∞ | λ = 0.75; Q/R = 0.01; Qf/R = 0.1; R = 1; γ★ = 0.554693; λ selected from the frozen six-point sweep and costs selected from the frozen 12-point grid on 50 disjoint GSM8K training prompts |
