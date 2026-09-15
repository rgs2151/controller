# Toxicity

## Goal

Reduce toxic model continuations without severely damaging output quality or general capability.

## Design

- **Base dataset:** RealToxicityPrompts (RTP).
- **Controller setup:** build and select controllers using only the RTP data reserved for those stages.
- **In-distribution evaluation:** generate continuations for 1,000 RTP prompts in each of five seeded repetitions.
- **Dataset transfer:** generate continuations for 1,000 Jigsaw prompts in each of five seeded repetitions using the RTP controller without refitting or recalibration.
- **Capability check:** answer the same fixed random sample of 200 five-shot MMLU questions used across all methods.

## Models

- Gemma-2-2B
- Llama-3-8B
- Qwen-2.5-14B

## Methods

- Original
- S-PID
- A-LQR
- H∞

## Measures

For RTP and Jigsaw:

- **Toxic continuations:** percentage classified as toxic.
- **Distinct-2:** diversity of generated word pairs.
- **Perplexity:** how unlikely the generated text is under the fixed evaluator model.

For MMLU:

- **MMLU accuracy:** percentage of the 200 questions answered with the correct A/B/C/D choice.

## Main comparison

Does a controller built on RTP reduce toxicity on RTP, transfer that reduction to Jigsaw, and preserve general MMLU performance?
