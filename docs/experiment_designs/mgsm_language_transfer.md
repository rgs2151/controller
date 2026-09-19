# MGSM multilingual language transfer

## Pipeline card

- **Status:** Qwen3-4B is complete; Llama-3.2-3B-Instruct is provisioned as the
  next run.
- **Task:** Solve MGSM arithmetic problems with native eight-shot examples.
- **Distribution shift:** Input language changes across Chinese, French, Japanese,
  Swahili, and Telugu.
- **Steered behavior:** Respond only in Spanish.
- **Direction data:** All 250 matched MGSM English/Spanish question pairs;
  paired Spanish-minus-English DiffMean.
- **Shared dynamics:** 50 frozen Spanish-question Jacobians; one `A` per model
  shared by A-LQR and H∞.
- **H∞ disturbance data:** 200 frozen GSM8K training questions.
- **Baseline settings:** Fixed S-PID and A-LQR settings; no sweep.
- **H∞ selection:** 50 disjoint GSM8K questions shared by every candidate;
  first select lambda at `Q/R=0.1`, `Qf/R=0.1`, `R=1`, then select from the
  12-point cost grid. Both stages maximize AXBench Overall.
- **Final evaluation:** 100 matched problems × 5 held-out languages;
  deterministic generation with a 256-token cap.
- **Models:** Qwen3-4B and Llama-3.2-3B-Instruct.
- **Methods:** Original, S-PID, A-LQR, and H∞ for both models.
- **Scoring:** Exact final-number accuracy, Spanish adherence, instruction
  relevance, fluency, and overall steering.
- **Evaluation size:** Per model, 900 H∞ selection generations and 2,000 final
  generations.

## Question

Can a controller make the model answer only in Spanish when the arithmetic
question itself moves into unseen input languages, while retaining correctness?

## Frozen scope

- **Models:** Qwen3-4B and Llama-3.2-3B-Instruct.
- **Methods:** Original, S-PID, A-LQR, and H∞.
- **Steering rule:** `respond only in Spanish, and no other language is allowed`.
- **Primary transfer languages:** Chinese, French, Japanese, Swahili, and Telugu.
- **Decoding:** deterministic, native eight-shot prompting, at most 256 new
  tokens, one generation per problem.
- **KV cache:** off.

## 1. Direction fitting

- Source: all 250 aligned MGSM test-question identities in English and Spanish.
- Undesired class: the English question.
- Desired class: the matched Spanish translation of the same question.
- Verify matching numerical answers before fitting.
- Estimator: paired, layer-wise Spanish-minus-English DiffMean over valid tokens.
- The texts are shared, but the direction is extracted separately in every model.

## 2. Shared dynamics

- Use the first frozen 50 Spanish direction questions.
- Average their prompt Jacobians into one saved `A` per model.
- A-LQR and H∞ use that exact matrix.

## 3. H∞ disturbance fitting

- Use 200 frozen GSM8K training questions outside MGSM evaluation.
- Fit `D`, reduced coordinates, and the base H∞ problem once per model.

## 4. Controller selection

- **S-PID:** fixed `lambda=1.5`, `Kp=0.5`, `Ki=0.5`, `Kd=0.01`.
- **A-LQR:** fixed `lambda=1.5`, `Q=0.1`, `R=1`, `Qf=0.1`. No sweep.
- **H∞ development set:** 50 additional GSM8K training questions, disjoint from
  the 200 disturbance prompts.
- **H∞ lambda stage:** sweep the configured lambda values while holding
  `Q/R=0.1`, `Qf/R=0.1`, and `R=1`; select maximum AXBench Overall, breaking an
  exact tie toward the smaller lambda. This stage is composition-configurable
  and may be disabled.
- **H∞ grid:** `R=1`, `Q/R in {0.01, 0.1, 1, 10}`, and
  `Qf/R in {0.01, 0.1, 0.316...}` using the selected lambda.
- **Objective:** mean of the per-response harmonic mean of AXBench concept
  relevance, instruction relevance, and fluency.

## 5. Final evaluation

- Use one fixed aligned subset of 100 problem identities.
- Evaluate the same identities in Chinese, French, Japanese, Swahili, and Telugu.
- English and Spanish build the steering direction and are not primary test
  conditions.
- No controller is refit by language.

## Scoring

- **Accuracy:** exact final-number match.
- **Spanish adherence:** deterministic AXBench rule score, 0 or 2.
- **Instruction relevance:** AXBench 0–2 rubric.
- **Fluency:** AXBench 0–2 rubric.
- **Overall steering:** per-response harmonic mean of the three steering scores.

## Evaluation size

- Completed Qwen: `100 × 5 languages × 4 methods = 2,000` generations.
- Planned Llama: `100 × 5 languages × 4 methods = 2,000` generations.
- H∞ selection per model: `6 × 50` lambda-stage generations followed by
  `12 × 50` cost-grid generations, for `900` short generations total.
- There are no repeated final-evaluation seeds.

## References

- [MGSM paper](https://arxiv.org/abs/2210.03057)
- [MGSM dataset](https://huggingface.co/datasets/juletxara/mgsm)
- [AXBench paper](https://arxiv.org/abs/2501.17148)
- [Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
