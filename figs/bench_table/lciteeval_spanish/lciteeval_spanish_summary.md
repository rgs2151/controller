# Spanish L-CiteEval language transfer — summary

| Model | Context | Method | Answer quality (%) ↑ | Citation F1 (%) ↑ | Overall steering (0–2) ↑ |
|---|---|---|---:|---:|---:|
| Llama-3.1-8B-Instruct | 8K | Original | 82.5 | 81.8 | 0.00 |
| Llama-3.1-8B-Instruct | 8K | A-LQR | 61.3 | 64.0 | 1.44 |
| Llama-3.1-8B-Instruct | 8K | H∞ (ours) | 60.0 | 59.9 | 1.52 |
| Llama-3.1-8B-Instruct | 16K | Original | 77.5 | 64.2 | 0.00 |
| Llama-3.1-8B-Instruct | 16K | A-LQR | 58.8 | 53.5 | 1.44 |
| Llama-3.1-8B-Instruct | 16K | H∞ (ours) | 55.0 | 54.6 | 1.31 |

## Method

- Task: answer the same 40 HotpotQA questions from numbered English evidence passages at approximately 8K and 16K tokens, cite the minimum supporting passages after every answer sentence, and produce the answer only in Spanish even though the evaluation prompt does not request Spanish.
- Dataset: `Jonaszky123/L-CiteEval`, pinned revision `c79c928529593f478e6573c969cf73d22f0cf0f9`, L-CiteEval-Length HotpotQA slice. Question identities and gold answers are matched across lengths.
- Direction: paired DiffMean over all 250 matched English–Spanish MGSM question pairs, fitted in Llama-3.1-8B's representation space. The shared dynamics matrix is the mean of 50 frozen Spanish-side prompt Jacobians and is used unchanged by A-LQR and H∞.
- H∞ disturbance fit: 200 disjoint upstream 2WikiMultihopQA training questions formatted as short L-Cite-style citation prompts.
- H∞ selection: 12 cost configurations evaluated on 10 frozen official L-CiteEval 2Wiki base-context questions. Selection maximizes the per-response harmonic mean of Spanish adherence, instruction relevance, and fluency.
- Generation: deterministic decoding, at most 200 new tokens, one generation per question, and evaluated-model KV cache disabled. The reported model is `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- Evaluation size: 40 questions × 2 matched context lengths × 3 methods = 240 generations. S-PID was deferred and can be appended later without changing these rows.

## Measures

| Column | Meaning | Scorer and decision rule |
|---|---|---|
| Answer quality (%) ↑ | Semantic correctness of the Spanish answer against the English question and official answer. | Bilingual `gpt-4o-mini-2024-07-18` rubric: 0 = incorrect or absent, 1 = partially correct, 2 = fully correct; divided by 2 and reported as a percentage. The response is never translated. |
| Citation recall (%) ↑ | Fraction of response claims jointly supported by their cited English passages. | Bilingual OpenAI entailment judge applies the original L-CiteEval/AutoAIS claim-level joint-entailment decision; code computes the original recall equation. |
| Citation precision (%) ↑ | Fraction of supplied citations judged necessary for supported claims. | The same bilingual judge performs independent-citation and leave-one-citation-out entailment tests; code computes the original precision equation. |
| Citation F1 (%) ↑ | Harmonic mean of citation recall and citation precision. | Computed deterministically per response from the two citation components, then averaged over 40 responses. |
| Spanish relevance (0–2) ↑ | Whether the response is written in Spanish. | Deterministic AXBench Spanish rule: 0 = not satisfied and 2 = satisfied. |
| Instruction relevance (0–2) ↑ | Whether the answer addresses the question and citation instruction. | AXBench instruction-relevance rubric through `gpt-4o-mini-2024-07-18`. |
| Fluency (0–2) ↑ | Language quality of the raw Spanish answer. | AXBench fluency rubric through `gpt-4o-mini-2024-07-18`. |
| Overall steering (0–2) ↑ | Joint Spanish adherence, instruction relevance, and fluency. | Per-response harmonic mean of the three AXBench scores; zero if any component is zero, then averaged over the 40 responses. |

These are descriptive means over one deterministic generation for each of 40 matched questions per context length, not repeated trials; therefore no standard errors are reported. Task-quality and steering-quality metrics are reported separately.

## Hyperparameters

| Model | Method | Hyperparameter choice |
|---|---|---|
| Llama-3.1-8B-Instruct | Original | No intervention |
| Llama-3.1-8B-Instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I; fixed without a sweep |
| Llama-3.1-8B-Instruct | H∞ | λ = 1.5; Q/R = 0.01; Qf/R = 0.01; R = 1; γ★ = 0.327605; selected by the frozen 10-question, 12-configuration calibration |
