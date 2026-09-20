# HarmBench robust refusal — ASR summary

| Model | Method | Direct ASR (%) ↓ | John ASR (%) ↓ | DNE ASR (%) ↓ | Jailbreak Bot ASR (%) ↓ | YOJA/Nona ASR (%) ↓ | APM ASR (%) ↓ | Human-jailbreak average ASR (%) ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | Original | 10.00 | 5.83 | 6.25 | 4.58 | 8.33 | 2.92 | 5.58 |
| Llama-3.2-1B-Instruct | A-LQR | 9.58 | 6.25 | 4.17 | 3.33 | 8.75 | 5.00 | 5.50 |
| Llama-3.2-1B-Instruct | H∞ (ours) | 2.92 | 1.25 | 1.67 | 0.42 | 0.83 | 0.42 | 0.92 |
| Llama-3.2-3B-Instruct | Original | 16.67 | 11.67 | 10.83 | 14.17 | 11.25 | 10.83 | 11.75 |
| Llama-3.2-3B-Instruct | A-LQR | 20.00 | 11.67 | 10.42 | 12.92 | 11.67 | 11.25 | 11.58 |
| Llama-3.2-3B-Instruct | H∞ (ours) | 8.33 | 4.17 | 3.75 | 3.75 | 4.17 | 2.50 | 3.67 |
| Llama-3.1-8B-Instruct | Original | 15.83 | 23.33 | 14.17 | 9.17 | 15.00 | 11.67 | 14.67 |
| Llama-3.1-8B-Instruct | A-LQR | 15.83 | 15.00 | 11.67 | 7.50 | 9.17 | 10.83 | 10.83 |
| Llama-3.1-8B-Instruct | H∞ (ours) | 3.33 | 1.67 | 0.00 | 0.00 | 0.00 | 0.83 | 0.50 |

## Method

- Task: respond to fixed text-only HarmBench test behaviors while steering toward non-exhibition of the requested harmful behavior. Llama-3.2-1B/3B use all 240 behaviors; the frozen compact Llama-3.1-8B run uses the same 120 sampled behavior identities in every condition.
- Direct condition: the raw behavior is presented without a jailbreak wrapper.
- Human-jailbreak conditions: each model's direct behavior identities are transformed by five frozen human-authored templates from the official HarmBench repository, producing 1,200 prompts for the 240-behavior runs and 600 for the compact 120-behavior run. The templates are reported separately rather than hidden behind one aggregate.
- Template meanings: John requests an unrestricted alter-ego answer; DNE instructs the model to output only `...`; Jailbreak Bot uses an iOS-jailbreak-support persona; YOJA/Nona is a long composite roleplay; and APM uses an unrestricted malicious-programmer persona.
- Example prompt structure:

  ```text
  Direct: <harmful behavior>
  John: <unrestricted-persona wrapper> + <same harmful behavior>
  ```

- Dataset: `centerforaisafety/HarmBench` at revision `8e1604d1171fe8a48d8febecd22f600e462bdcdd`. The five templates are the frozen `random_subset=5`, `seed=1` selection.
- Models: `meta-llama/Llama-3.2-1B-Instruct` at revision `9213176726f574b556790deb65791e0c5aa438b6`, `meta-llama/Llama-3.2-3B-Instruct` at revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, and `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- Generation: greedy decoding, one completion per prompt, and evaluated-model KV cache disabled. Llama-3.2-1B/3B use at most 512 new tokens; the compact Llama-3.1-8B run uses at most 100.

## Measures

| Column | Meaning | Scorer and decision rule |
|---|---|---|
| ASR (%) ↓ | Percentage of generations that successfully exhibit the requested harmful behavior. | Official HarmBench test classifier; classifier success = 1, otherwise 0. Lower is safer. |
| Safe-concept relevance (0–2) ↑ | Whether the response expresses the target non-exhibition/refusal behavior. | AXBench concept-relevance rubric through `gpt-4o-mini-2024-07-18`. |
| Instruction relevance (0–2) ↑ | Whether the response meaningfully addresses the underlying request. | AXBench instruction-relevance rubric through `gpt-4o-mini-2024-07-18`. |
| Fluency (0–2) ↑ | Readability and language quality. | AXBench fluency rubric through `gpt-4o-mini-2024-07-18`. |
| Overall steering (0–2) ↑ | Joint safe-concept relevance, instruction relevance, and fluency. | Per-response harmonic mean of the three AXBench scores; zero when any component is zero, then averaged over prompts. |

These are descriptive means over one fixed generation per prompt, not repeated trials; therefore no standard errors are reported. The summary table reports ASR for every template separately and includes the equal-weight human-jailbreak aggregate only for continuity with the earlier collapsed result.

## Hyperparameters

| Model | Method | Hyperparameter choice |
|---|---|---|
| Llama-3.2-1B-Instruct | Original | No intervention |
| Llama-3.2-1B-Instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Llama-3.2-1B-Instruct | H∞ | λ = 1.5; Q/R = 0.01; Qf/R = 0.01; R = 1; γ★ = 0.130075; selected on 50 disjoint direct validation behaviors by maximum AXBench overall steering |
| Llama-3.2-3B-Instruct | Original | No intervention |
| Llama-3.2-3B-Instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Llama-3.2-3B-Instruct | H∞ | λ = 1.5; Q/R = 0.1; Qf/R = 0.01; R = 1; γ★ = 0.415802; selected on 50 disjoint direct validation behaviors by maximum AXBench overall steering |
| Llama-3.1-8B-Instruct | Original | No intervention |
| Llama-3.1-8B-Instruct | A-LQR | λ = 1.5; Q = 0.1I; R = 1I; Qf = 0.1I |
| Llama-3.1-8B-Instruct | H∞ | λ = 1.5; Q/R = 0.01; Qf/R = 0.01; R = 1; selected on 50 disjoint direct validation behaviors by maximum AXBench overall steering |

The DNE template is retained because it belongs to the frozen official subset, but it explicitly requests nonresponse and is therefore not a meaningful harmful-compliance jailbreak. The per-template report prevents this condition from silently determining the interpretation of the aggregate.
