# MGSM multilingual transfer — overall

| Model | Method | Overall steering (0–2) ↑ | Exact answer (%) ↑ | Spanish rule (0–2) ↑ | Instruction relevance (0–2) ↑ | Fluency (0–2) ↑ |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | Original | 0.000 | 41.6 | 0.00 | 1.76 | 1.47 |
| Qwen3-4B | S-PID | 0.000 | 0.0 | 1.98 | 0.00 | 0.00 |
| Qwen3-4B | A-LQR | 1.270 | 22.4 | 1.81 | 1.51 | 1.08 |
| Qwen3-4B | H∞ (ours) | 1.141 | 32.2 | 1.44 | 1.66 | 1.25 |

## Method

- Task: solve the same 250 MGSM arithmetic problems in Chinese, French, Japanese, Swahili, and Telugu while steering every response toward Spanish. English and Spanish are excluded from evaluation.
- Direction: all 250 matched English–Spanish MGSM pairs define the Spanish steering direction. A-LQR and H∞ share the same 50-Jacobian dynamics estimate. H∞ additionally fits its disturbance geometry and robust controller without changing the shared dynamics matrix.
- Prompting: each language uses its native eight-shot worked-example prompt. Generation is deterministic, limited to 256 new tokens, and runs with evaluated-model KV cache disabled.
- Model: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`, with thinking mode disabled.
- The overall report is the equal-weight macro-average of the five language-level means. The full report exposes all 20 language–method cells.

## Measures

| Column | Meaning | Scorer and decision rule |
|---|---|---|
| Overall steering (0–2) ↑ | Joint Spanish adherence, instruction relevance, and fluency. | Per-response harmonic mean of the three AXBench components; zero if any component is zero, then averaged over the 250 problems. |
| Exact answer (%) ↑ | Percentage of problems with the correct final numeric answer. | Deterministic language-independent parser; commas and a trailing `.0` are normalized, and missing or unparseable answers are incorrect. |
| Spanish rule (0–2) ↑ | Whether the generated response is in Spanish. | AXBench deterministic Spanish rule evaluator: 0 = rule not satisfied and 2 = rule satisfied. |
| Instruction relevance (0–2) ↑ | Whether the response addresses and attempts the arithmetic task. | AXBench instruction-relevance rubric through `gpt-4o-mini-2024-07-18`; integer score 0, 1, or 2. |
| Fluency (0–2) ↑ | Language quality of the generated response. | AXBench fluency rubric through `gpt-4o-mini-2024-07-18`; integer score 0, 1, or 2. |

These are descriptive means on one fixed 250-problem evaluation set per language, not repeated trials; therefore the table does not report standard errors. Every method uses the same problem identities within a language.

## Hyperparameters

| Model | Method | Hyperparameter choice |
|---|---|---|
| Qwen3-4B | Original | No intervention |
| Qwen3-4B | S-PID | λ = 1.5; Kp = 0.5; Ki = 0.5; Kd = 0.01 |
| Qwen3-4B | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Qwen3-4B | H∞ | λ = 1.5; Q/R = 0.01; Qf/R = 0.316227766; R = 1; γ★ = 11.0736; selected on 50 disjoint GSM8K training prompts |
