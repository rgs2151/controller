# L-CiteEval length transfer — summary

| Context | Model | Method | Answer recall (%) ↑ | Citation F1 (%) ↑ | Overall steering (0–2) ↑ |
|---|---|---|---:|---:|---:|
| 8K | Qwen2.5-3B-Instruct | Original | 66.2 | 7.8 | 0.00 |
| 8K | Qwen2.5-3B-Instruct | S-PID | 72.1 | 7.2 | 0.00 |
| 8K | Qwen2.5-3B-Instruct | A-LQR | 70.1 | 5.6 | 0.05 |
| 8K | Qwen2.5-3B-Instruct | H∞ (ours) | 66.4 | 6.4 | 0.00 |
| 16K | Qwen2.5-3B-Instruct | Original | 65.2 | 5.8 | 0.15 |
| 16K | Qwen2.5-3B-Instruct | S-PID | 61.9 | 7.2 | 0.00 |
| 16K | Qwen2.5-3B-Instruct | A-LQR | 64.8 | 3.0 | 0.00 |
| 16K | Qwen2.5-3B-Instruct | H∞ (ours) | 65.2 | 7.8 | 0.10 |

## Method

- Task: answer the same 40 HotpotQA questions from numbered evidence passages at approximately 8K and 16K tokens, citing the minimum supporting passages after every answer sentence.
- Dataset: `Jonaszky123/L-CiteEval`, pinned revision `c79c928529593f478e6573c969cf73d22f0cf0f9`, L-CiteEval-Length HotpotQA slice. The 40 question identities and gold answers are matched across both context lengths.
- Steering concept: AXBench concept 499, `positive sentiments and descriptions of enjoyable experiences`, using all 72 released positive responses and 72 genre-matched negative responses.
- Controllers: A-LQR and H∞ share the same saved 50-Jacobian dynamics estimate. H∞ separately fits its 200-sample disturbance geometry and robust controller.
- Generation: official one-shot HotpotQA prompt, deterministic decoding, at most 200 new tokens, and evaluated-model KV cache disabled for every method.
- Model: `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, using the same static YaRN configuration at both lengths.
- The 8K and 16K conditions remain separate; no cross-length average is reported.

## Measures

| Column | Meaning | Scorer and decision rule |
|---|---|---|
| Answer recall (%) ↑ | Gold-answer tokens recovered by the generated answer. | Official normalized L-CiteEval token-overlap recall after removing citation markers; the best matching released gold answer is used. |
| Citation F1 (%) ↑ | Balance between supported claims and necessary citations. | Pinned `tasksource/deberta-base-long-nli` at revision `04dcf11f844b07bc57015169fca2b7d6df8299d5`, applied only to each claim and its cited passages. |
| Overall steering (0–2) ↑ | Joint target-concept presence, instruction relevance, and fluency. | Per-response harmonic mean of the three AXBench 0–2 scores; zero if any component is zero, then averaged over the 40 responses. |

These are descriptive means on one deterministic generation for each of 40 matched questions per context length, not repeated trials; therefore the table does not report standard errors.

## Hyperparameters

| Model | Method | Hyperparameter choice |
|---|---|---|
| Qwen2.5-3B-Instruct | Original | No intervention |
| Qwen2.5-3B-Instruct | S-PID | λ = 1.5; Kp = 0.5; Ki = 0.5; Kd = 0.01; frozen upstream concept-steering configuration |
| Qwen2.5-3B-Instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I; frozen upstream concept-steering configuration |
| Qwen2.5-3B-Instruct | H∞ | λ = 1.5; Q/R = 0.01; Qf/R = 0.01; R = 1; γ★ = 2.3054; selected on 50 disjoint short AXBench-style prompts |
