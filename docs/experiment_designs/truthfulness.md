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
- **H∞ selection:** 50 further disjoint prompts; 32 cost configurations; maximize
  AXBench overall steering.
- **Final evaluation:** 817 questions × 5 seeds in English and Spanish; optional
  fixed 200-question five-shot MMLU.
- **Models:** Gemma-2-2B, Llama-3-8B, and Qwen-2.5-14B.
- **Methods:** Original, ITI, ActAdd, Mean-AcT, Linear-AcT, PID-AcT, ODESteer,
  S-PID, A-LQR, and H∞.
- **Scoring:** True, Informative, concept relevance, instruction relevance,
  fluency, AXBench overall, and optional MMLU accuracy.
- **Evaluation size:** Per model, 1,600 H∞ selection generations and 81,700 final
  TruthfulQA generations when all methods and both languages are run.

## Question

Can a controller increase truthful answers on TruthfulQA and retain that behavior
when the same questions are translated into Spanish?

## Frozen scope

- **Models:** Gemma-2-2B, Llama-3-8B, and Qwen-2.5-14B.
- **Methods:** Original, ITI, ActAdd, Mean-AcT, Linear-AcT, PID-AcT,
  ODESteer, S-PID, A-LQR, and H∞.
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

- **A-LQR:** use the published model-specific setting; do not sweep it.
- **Other baselines:** use their frozen published or project settings; do not
  sweep them on final TruthfulQA.
- **H∞ development set:** 50 additional TruthfulQA prompts, disjoint from all fit
  records and final evaluation prompts.
- **H∞ grid:** `R=1`, eight frozen `Q/R` values, and four frozen `Qf/R` values.
- **Objective:** maximum mean AXBench overall steering, the harmonic mean of
  truthful-concept relevance, instruction relevance, and fluency.
- True and Informative are final outcomes, not selection objectives.

## 5. Final evaluation

- **English:** all 817 TruthfulQA questions × 5 seeded repetitions.
- **Spanish transfer:** Spanish translations of the same 817 questions × 5
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

- Per dataset and method: `817 × 5 = 4,085` generations.
- Spanish is a language transfer condition, not a new calibration condition.
