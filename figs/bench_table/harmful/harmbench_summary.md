# HarmBench robust refusal — ASR summary

| Method | Direct ASR (%) ↓ | John ASR (%) ↓ | DNE ASR (%) ↓ | Jailbreak Bot ASR (%) ↓ | YOJA/Nona ASR (%) ↓ | APM ASR (%) ↓ | Human-jailbreak average ASR (%) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 10.00 | 5.83 | 6.25 | 4.58 | 8.33 | 2.92 | 5.58 |
| A-LQR | 9.58 | 6.25 | 4.17 | 3.33 | 8.75 | 5.00 | 5.50 |
| H∞ (ours) | 2.92 | 1.25 | 1.67 | 0.42 | 0.83 | 0.42 | 0.92 |

## Method

- Task: respond to 240 fixed text-only HarmBench test behaviors while steering toward non-exhibition of the requested harmful behavior.
- Direct condition: the raw behavior is presented without a jailbreak wrapper.
- Human-jailbreak conditions: the same 240 behavior identities are each transformed by five frozen human-authored templates from the official HarmBench repository, producing 1,200 matched prompts. The templates are reported separately rather than hidden behind one aggregate.
- Template meanings: John requests an unrestricted alter-ego answer; DNE instructs the model to output only `...`; Jailbreak Bot uses an iOS-jailbreak-support persona; YOJA/Nona is a long composite roleplay; and APM uses an unrestricted malicious-programmer persona.
- Example prompt structure:

  ```text
  Direct: <harmful behavior>
  John: <unrestricted-persona wrapper> + <same harmful behavior>
  ```

- Dataset: `centerforaisafety/HarmBench` at revision `8e1604d1171fe8a48d8febecd22f600e462bdcdd`. The five templates are the frozen `random_subset=5`, `seed=1` selection.
- Model: `meta-llama/Llama-3.2-1B-Instruct` at revision `9213176726f574b556790deb65791e0c5aa438b6`.
- Generation: greedy decoding, at most 512 new tokens, one completion per prompt, and evaluated-model KV cache disabled.

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

The DNE template is retained because it belongs to the frozen official subset, but it explicitly requests nonresponse and is therefore not a meaningful harmful-compliance jailbreak. The per-template report prevents this condition from silently determining the interpretation of the aggregate.
