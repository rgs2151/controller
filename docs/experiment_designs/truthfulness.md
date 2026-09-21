# Truthfulness

## Pipeline card

- **Status:** Implemented; reported runs use KV cache off.
- **Task:** Answer open-ended TruthfulQA questions.
- **Distribution shift:** English questions are translated into Spanish while
  answers remain requested in English.
- **Steered behavior:** Truthful, factually accurate, non-misleading answers.
- **Direction data:** 200 false-answer strings and 200 true-answer strings from
  TruthfulQA; layer-wise DiffMean.
- **Shared dynamics:** 35 independent true-answer Jacobians; one `A` per model
  shared by A-LQR and H∞.
- **H∞ disturbance data:** 200 disjoint TruthfulQA prompts.
- **Baseline settings:** Published or frozen project settings; no baseline sweep.
- **H∞ selection:** Fix the setpoint multiplier to the same frozen value used by
  A-LQR and select among 32 `Q/R` and `Qf/R` configurations on 200 further
  disjoint prompts by aggregate True percentage times Informative percentage.
- **Final evaluation:** 817 questions × 1 seed in English and Spanish; optional
  fixed 200-question five-shot MMLU.
- **Models:** Gemma-2-2B, Llama-3-8B, Qwen-2.5-14B, Qwen-2.5-32B, GPT-2 XL,
  and the prepared Qwen-2.5-0.5B fallback.
- **Methods:** Original, A-LQR, and H∞ by default; additional registered methods
  remain available for legacy comparisons.
- **Scoring:** True, Informative, concept relevance, instruction relevance,
  fluency, AXBench overall, and optional MMLU accuracy.
- **Evaluation size:** Per new model, 6,400 H∞ selection generations and 4,902
  final TruthfulQA generations for three methods and both languages.

## Question

Can a controller increase truthful answers on TruthfulQA and retain that behavior
when the same questions are translated into Spanish?

## Frozen scope

- **Models:** Gemma-2-2B, Llama-3-8B, Qwen-2.5-14B, Qwen-2.5-32B, GPT-2 XL,
  and Qwen-2.5-0.5B.
- **Default methods:** Original, A-LQR, and H∞.
- **KV cache:** off for reported runs.
- **Base task:** open-ended TruthfulQA.

## 1. Direction fitting

- Source: TruthfulQA multiple-choice validation data.
- Undesired class: 200 question–false-answer strings.
- Desired class: 200 question–true-answer strings.
- Estimator: layer-wise desired-minus-undesired DiffMean.
- The saved direction and setpoint are fitted separately for every model.

## 2. Shared dynamics

- Use 35 independently sampled true-answer strings.
- Average their prompt Jacobians into one saved nominal matrix `A`.
- A-LQR and H∞ use the same `A` artifact.

## 3. H∞ disturbance fitting

- Use 200 TruthfulQA prompts disjoint from the direction and Jacobian records.
- Fit the reduced coordinates and disturbance channels `D` once.
- Recompute the H∞ solution for each cost configuration; preserve `gamma*` and
  Hannah's diagnostic bundle for the selected controller.

## 4. Controller selection

- **A-LQR:** use the published setting where available; GPT-2 XL and
  Qwen-2.5-0.5B reuse the frozen small-model project baseline (`lambda=3`,
  `Q=0.1`, `R=1`, `Qf=0.3`) with no sweep.
- **Other baselines:** use their frozen published or project settings; do not
  sweep them on final TruthfulQA.
- **H∞ development set:** 200 additional TruthfulQA prompts, disjoint from all
  direction, Jacobian, and disturbance-fit records.
- **H∞ setpoint multiplier:** fixed to the same frozen model-specific value used
  by A-LQR; it is not swept.
- **H∞ grid:** `R=1`, eight frozen `Q/R` values, and four frozen `Qf/R` values.
- **Default objective:** aggregate `mean(True) × mean(Informative)`; this is the
  historical T×I multiplication, not a per-response intersection rate.
- **Registered alternatives:** T×I with fluency, historical mean True percentage,
  mean AXBench overall, and the prior Truthfulness quality composite.

## 5. Final evaluation

- **English:** all 817 TruthfulQA questions × 1 seeded repetition.
- **Spanish transfer:** Spanish translations of the same 817 questions × 1
  seeded repetitions. The prompt requests an English answer.
- The English-fitted controller is reused without refitting or reselection.
- **Optional capability check:** one frozen 200-question, five-shot MMLU sample.

## Scoring

- **True:** pinned TruthfulQA truth judge; binary percentage.
- **Informative:** pinned TruthfulQA helpfulness judge; binary percentage.
- **Concept relevance:** AXBench 0–2 truthfulness-concept rubric.
- **Instruction relevance:** AXBench 0–2 rubric.
- **Fluency:** AXBench 0–2 rubric.
- **AXBench overall:** per-response harmonic mean of the preceding three AXBench
  scores.
- **MMLU accuracy:** exact A/B/C/D accuracy; no learned judge.

## Evaluation size

- Per dataset and method: `817 × 1 = 817` generations.
- Spanish is a language transfer condition, not a new calibration condition.
