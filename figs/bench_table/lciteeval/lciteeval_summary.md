# L-CiteEval length transfer — summary

| Model | Context | Method | Answer recall (%) ↑ | Citation F1 (%) ↑ | Overall steering (0–2) ↑ |
|---|---|---|---:|---:|---:|
| Qwen2.5-3B-Instruct | 8K | Original | 66.2 ± 8.1 | 7.8 ± 1.6 | 0.00 ± 0.00 |
| Qwen2.5-3B-Instruct | 8K | S-PID | 72.1 ± 9.3 | 7.2 ± 2.1 | 0.00 ± 0.00 |
| Qwen2.5-3B-Instruct | 8K | A-LQR | 70.1 ± 8.9 | 5.6 ± 2.0 | 0.05 ± 0.05 |
| Qwen2.5-3B-Instruct | 8K | H∞ (ours) | 66.4 ± 7.9 | 6.4 ± 2.0 | 0.00 ± 0.00 |
| Qwen2.5-3B-Instruct | 16K | Original | 65.2 ± 7.6 | 5.8 ± 2.2 | 0.15 ± 0.08 |
| Qwen2.5-3B-Instruct | 16K | S-PID | 61.9 ± 10.4 | 7.2 ± 3.2 | 0.00 ± 0.00 |
| Qwen2.5-3B-Instruct | 16K | A-LQR | 64.8 ± 9.8 | 3.0 ± 1.5 | 0.00 ± 0.00 |
| Qwen2.5-3B-Instruct | 16K | H∞ (ours) | 65.2 ± 8.6 | 7.8 ± 3.6 | 0.10 ± 0.07 |
| Qwen2.5-3B-Instruct | 32K | Original | 54.0 ± 8.4 | 2.4 ± 1.3 | 0.00 ± 0.00 |
| Qwen2.5-3B-Instruct | 32K | S-PID | 61.0 ± 9.1 | 1.6 ± 1.1 | 0.00 ± 0.00 |
| Qwen2.5-3B-Instruct | 32K | A-LQR | 56.2 ± 6.7 | 4.1 ± 1.9 | 0.00 ± 0.00 |
| Qwen2.5-3B-Instruct | 32K | H∞ (ours) | 49.2 ± 6.1 | 4.3 ± 2.4 | 0.00 ± 0.00 |
| Llama-3.2-1B-Instruct | 8K | Original | 58.5 ± 7.7 | 6.2 ± 2.5 | 0.04 ± 0.04 |
| Llama-3.2-1B-Instruct | 8K | A-LQR | 59.8 ± 9.3 | 6.6 ± 2.7 | 0.04 ± 0.04 |
| Llama-3.2-1B-Instruct | 8K | H∞ (ours) | 61.7 ± 7.8 | 6.4 ± 2.9 | 0.04 ± 0.04 |

## Method

- Task: answer the same 40 HotpotQA questions from numbered evidence passages at approximately 8K, 16K, and 32K tokens, citing the minimum supporting passages after every answer sentence.
- Dataset: `Jonaszky123/L-CiteEval`, pinned revision `c79c928529593f478e6573c969cf73d22f0cf0f9`, L-CiteEval-Length HotpotQA slice. The 40 question identities and gold answers are matched across all three context lengths.
- Steering concept: AXBench concept 499, `positive sentiments and descriptions of enjoyable experiences`, using all 72 released positive responses and 72 genre-matched negative responses.
- Controllers: A-LQR and H∞ share the same saved 50-Jacobian dynamics estimate. H∞ separately fits its 200-sample disturbance geometry and robust controller.
- Generation: official one-shot HotpotQA prompt, deterministic decoding, at most 200 new tokens, and evaluated-model KV cache disabled for every method.
- Models: `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, using the same static YaRN configuration at all three lengths, and `meta-llama/Llama-3.2-1B-Instruct` at revision `9213176726f574b556790deb65791e0c5aa438b6` at 8K.
- Context conditions remain separate; no cross-length or cross-model average is reported.

## Measures

| Column | Meaning | Scorer and decision rule |
|---|---|---|
| Answer recall (%) ↑ | Gold-answer tokens recovered by the generated answer. | Official normalized L-CiteEval token-overlap recall after removing citation markers; the best matching released gold answer is used. |
| Citation F1 (%) ↑ | Balance between supported claims and necessary citations. | Pinned `tasksource/deberta-base-long-nli` at revision `04dcf11f844b07bc57015169fca2b7d6df8299d5`, applied only to each claim and its cited passages. |
| Overall steering (0–2) ↑ | Joint target-concept presence, instruction relevance, and fluency. | Per-response harmonic mean of the three AXBench 0–2 scores; zero if any component is zero, then averaged over the 40 responses. |

Values are full-sample means ± ten-group delete-one-group jackknife standard errors. Each group contains four matched HotpotQA question identities; the same partition is used at 8K, 16K, and 32K. The uncertainty measures question-sampling variability, not decoding-run or judge variability.

## Hyperparameters

| Model | Method | Hyperparameter choice |
|---|---|---|
| Qwen2.5-3B-Instruct | Original | No intervention |
| Qwen2.5-3B-Instruct | S-PID | λ = 1.5; Kp = 0.5; Ki = 0.5; Kd = 0.01; frozen upstream concept-steering configuration |
| Qwen2.5-3B-Instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I; frozen upstream concept-steering configuration |
| Qwen2.5-3B-Instruct | H∞ | 8K/16K: λ = 1.5; Q/R = 0.01; Qf/R = 0.01; R = 1; γ★ = 2.3054. Updated 32K: λ = 1.5; Q/R = 0.1; Qf/R = 0.01; R = 1; γ★ = 2.9917; selected on the 40 matched 8K HotpotQA prompts with 45% answer recall, 45% citation F1, 5% concept relevance, and 5% fluency. |
| Llama-3.2-1B-Instruct | Original | No intervention |
| Llama-3.2-1B-Instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Llama-3.2-1B-Instruct | H∞ | λ = 1.5; Q/R = 0.01; Qf/R = 0.01; R = 1; γ★ = 0.08690; selected on the 40 8K HotpotQA prompts with 45% answer recall, 45% citation F1, 5% concept relevance, and 5% fluency; no A-LQR fallback was used. |
