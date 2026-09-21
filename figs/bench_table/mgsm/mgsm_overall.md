# MGSM multilingual transfer — summary

| Model | Method | Accuracy (%) ↑ | Spanish relevance (0–2) ↑ | Instruction relevance (0–2) ↑ | Fluency (0–2) ↑ |
|---|---|---:|---:|---:|---:|
| Qwen3-4B | Original | 41.6 ± 2.5 | 0.00 ± 0.00 | 1.76 ± 0.03 | 1.47 ± 0.03 |
| Qwen3-4B | A-LQR | 22.4 ± 2.3 | 1.81 ± 0.02 | 1.51 ± 0.04 | 1.08 ± 0.02 |
| Qwen3-4B | H∞ (ours) | 32.2 ± 2.4 | 1.44 ± 0.03 | 1.66 ± 0.03 | 1.25 ± 0.04 |
| Llama-3.2-3B | Original | 33.0 ± 1.6 | 0.00 ± 0.00 | 1.68 ± 0.03 | 1.44 ± 0.03 |
| Llama-3.2-3B | A-LQR | 27.6 ± 2.6 | 1.50 ± 0.03 | 1.57 ± 0.04 | 1.17 ± 0.04 |
| Llama-3.2-3B | H∞ (ours) | 12.2 ± 2.4 | 1.47 ± 0.02 | 1.02 ± 0.05 | 0.91 ± 0.04 |
| Phi-4-mini | Original | 45.4 ± 2.6 | 0.00 ± 0.00 | 1.79 ± 0.02 | 1.52 ± 0.04 |
| Phi-4-mini | A-LQR | 0.0 ± 0.0 | 1.47 ± 0.04 | 0.01 ± 0.01 | 0.02 ± 0.01 |
| Phi-4-mini | H∞ (ours) | 35.2 ± 2.6 | 1.21 ± 0.04 | 1.56 ± 0.04 | 1.05 ± 0.03 |

## Method

- Task: solve matched MGSM arithmetic problems in Chinese, French, Japanese, Swahili, and Telugu while steering every response toward Spanish. English and Spanish are excluded from evaluation.
- Direction: all 250 matched English–Spanish MGSM pairs define the Spanish steering direction. A-LQR and H∞ share the same 50-Jacobian dynamics estimate. H∞ additionally fits its disturbance geometry and robust controller without changing the shared dynamics matrix.
- Prompting: each language uses its native eight-shot worked-example prompt. Generation is deterministic, limited to 256 new tokens, and runs with evaluated-model KV cache disabled.
- Models: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c` with thinking mode disabled, `meta-llama/Llama-3.2-3B-Instruct` at revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, and `microsoft/Phi-4-mini-instruct` at revision `cfbefacb99257ffa30c83adab238a50856ac3083`.
- Evaluation size: all three models use the same frozen 100-problem subset per language. The active comparison reports Original, A-LQR, and H∞ on identical problem identities within each model. The summary macro-averages the five language means independently within each model.

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
| Qwen3-4B | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Qwen3-4B | H∞ | λ = 1.5; Q/R = 0.01; Qf/R = 0.316227766; R = 1; γ★ = 11.0736; selected on 50 disjoint GSM8K training prompts |
| Llama-3.2-3B-Instruct | Original | No intervention |
| Llama-3.2-3B-Instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Llama-3.2-3B-Instruct | H∞ | λ = 1.5 fixed; Q/R = 1; Qf/R = 0.01; R = 1; γ★ = 1.919061; costs selected from the frozen 12-point grid on 50 disjoint translated GSM8K training prompts balanced across Bengali, German, Russian, and Thai using 80% exact-answer accuracy and 20% normalized AXBench Overall; no λ sweep |
| Phi-4-mini-instruct | Original | No intervention |
| Phi-4-mini-instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Phi-4-mini-instruct | H∞ | λ = 1.5 fixed; Q/R = 0.1; Qf/R = 0.01; R = 1; γ★ = 4.976749; costs selected from the frozen 12-point grid on 100 disjoint translated GSM8K training prompts balanced across Bengali, German, Russian, and Thai using 90% exact-answer accuracy and 10% normalized AXBench Overall; no λ sweep |
