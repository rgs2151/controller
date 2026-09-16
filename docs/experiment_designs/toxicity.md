# Toxicity

## Pipeline card

- **Status:** Implemented; the current Gemma-2-2B run uses KV cache off.
- **Task:** Continue RealToxicityPrompts text.
- **Distribution shift:** None in the default run; MMLU is an optional capability
  evaluation, not a toxicity transfer set.
- **Steered behavior:** Non-toxic, respectful, non-abusive language.
- **Direction data:** 200 high-toxicity and 200 low-toxicity RTP prompts;
  layer-wise DiffMean.
- **Shared dynamics:** 50 independent low-toxicity RTP Jacobians; one `A` per
  model shared by A-LQR and H∞.
- **H∞ disturbance data:** 200 disjoint RTP prompts.
- **Baseline settings:** Fixed S-PID and published A-LQR settings; no sweep.
- **H∞ selection:** 50 further disjoint RTP prompts; 12 cost configurations;
  maximize AXBench overall steering.
- **Final evaluation:** 1,000 RTP prompts × 5 seeds; optional fixed 200-question
  five-shot MMLU.
- **Models:** Gemma-2-2B.
- **Methods:** Original, S-PID, A-LQR, and H∞.
- **Scoring:** Toxicity percentage, Distinct-2, perplexity, concept relevance,
  instruction relevance, fluency, AXBench overall, and optional MMLU accuracy.
- **Evaluation size:** 600 H∞ selection generations and 20,000 final RTP
  generations.

## Question

Can a controller reduce toxic continuations on RealToxicityPrompts without
destroying diversity, fluency, or general capability?

## Frozen scope

- **Model:** Gemma-2-2B.
- **Methods:** Original, S-PID, A-LQR, and H∞.
- **Base task:** RealToxicityPrompts (RTP).
- **KV cache:** off for reported runs.
- **Removed:** Jigsaw transfer is not part of the current design.
- **Optional:** MMLU is composable but not part of the default run.

## 1. Direction fitting

- Undesired class: 200 RTP prompts with source toxicity in `[0.8, 1.0]`.
- Desired class: 200 RTP prompts with source toxicity in `[0.0, 0.1]`.
- Estimator: layer-wise desired-minus-undesired DiffMean.

## 2. Shared dynamics

- Use 50 independently sampled low-toxicity RTP prompts.
- Average their prompt Jacobians into one saved nominal matrix `A`.
- A-LQR and H∞ use the same `A` artifact.

## 3. H∞ disturbance fitting

- Use 200 RTP prompts disjoint from direction fitting and Jacobian fitting.
- Fit reduced coordinates and disturbance channels `D` once.
- Preserve the selected controller's `gamma*` and Hannah diagnostic bundle.

## 4. Controller selection

- **S-PID:** fixed at `lambda=1`, `Kp=0.7`, `Ki=0.01`, `Kd=0.1`.
- **A-LQR:** published Gemma toxicity setting: `lambda=3.5`, `Q=0.1`,
  `R=1`, `Qf=0.1`. No sweep.
- **H∞ development set:** 50 additional disjoint RTP prompts, one generation
  per candidate.
- **H∞ grid:** `R=1`, `Q/R in {0.01, 0.1, 1, 10}`, and
  `Qf/R in {0.01, 0.1, 0.316...}`.
- **Objective:** maximum mean AXBench overall steering, the harmonic mean of
  non-toxic-concept relevance, instruction relevance, and fluency.

## 5. Final evaluation

- 1,000 frozen RTP prompts × 5 seeded repetitions per method.
- Fit, development, and final prompt IDs are disjoint. Repetitions reuse the
  same evaluation pool with different generation seeds.
- Optional MMLU uses one frozen set of 200 five-shot questions.

## Scoring

- **Toxicity:** percentage classified as toxic by the pinned classifier.
- **Distinct-2:** generated bigram diversity.
- **Perplexity:** fixed evaluator-model perplexity.
- **Concept relevance, instruction relevance, fluency:** AXBench 0–2 scores.
- **AXBench overall:** their per-response harmonic mean.
- **MMLU accuracy:** exact A/B/C/D accuracy when the optional dataset is run.

## Evaluation size

- Default RTP evaluation: `1,000 × 5 × 4 methods = 20,000` generations.
- H∞ selection: `12 × 50 = 600` short generations.
