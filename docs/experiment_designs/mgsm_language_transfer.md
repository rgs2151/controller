# MGSM multilingual language transfer

## Pipeline card

- **Status:** Qwen3-4B and Llama-3.2-3B-Instruct are complete;
  Phi-4-mini-instruct and Granite-3.3-2B-Instruct are registered model
  expansions.
- **Task:** Solve MGSM arithmetic problems with native eight-shot examples.
- **Distribution shift:** Input language changes across Chinese, French, Japanese,
  Swahili, and Telugu.
- **Steered behavior:** Respond only in Spanish.
- **Direction data:** All 250 matched MGSM English/Spanish question pairs;
  paired Spanish-minus-English DiffMean.
- **Shared dynamics:** 50 frozen Spanish-question Jacobians; one `A` per model
  shared by A-LQR and H∞.
- **H∞ disturbance data:** 200 frozen GSM8K training questions translated and
  balanced across Bengali, German, Russian, and Thai.
- **Baseline settings:** Fixed A-LQR settings; no sweep.
- **H∞ selection:** 100 additional translated GSM8K questions balanced across
  Bengali, German, Russian, and Thai and shared by every candidate; `lambda=1.5`
  is fixed to the same target as A-LQR, and the 12-point cost grid maximizes the
  per-response accuracy-weighted additive score: 90% exact-answer accuracy and
  10% normalized AXBench Overall.
- **Final evaluation:** 100 matched problems × 5 held-out languages;
  deterministic generation with a 256-token cap.
- **Models:** Qwen3-4B, Llama-3.2-3B-Instruct, Phi-4-mini-instruct, and
  Granite-3.3-2B-Instruct.
- **Methods:** Original, A-LQR, and H∞ for every active model run. S-PID is
  retained in code only as an optional deferred method.
- **Scoring:** Exact final-number accuracy, Spanish adherence, instruction
  relevance, fluency, and overall steering.
- **Evaluation size:** Per model, 1,200 H∞ selection generations and 1,500 final
  generations.

## Question

Can a controller make the model answer only in Spanish when the arithmetic
question itself moves into unseen input languages, while retaining correctness?

## Frozen scope

- **Models:** Qwen3-4B, Llama-3.2-3B-Instruct, Phi-4-mini-instruct, and
  Granite-3.3-2B-Instruct.
- **Methods:** Original, A-LQR, and H∞.
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

- Use 200 frozen GSM8K training questions outside MGSM evaluation, with 50
  translated into each of Bengali, German, Russian, and Thai.
- Wrap every translated question in that language's native MGSM eight-shot
  chain-of-thought demonstrations.
- Fit `D`, reduced coordinates, and the base H∞ problem once per model.

## 4. Controller selection

- **A-LQR:** fixed `lambda=1.5`, `Q=0.1`, `R=1`, `Qf=0.1`. No sweep.
- **H∞ development set:** 100 additional GSM8K training questions, disjoint from
  the 200 disturbance prompts: 25 each in Bengali, German, Russian, and Thai.
- **Isolation:** Bengali, German, Russian, and Thai are calibration-only. Chinese,
  French, Japanese, Swahili, and Telugu remain untouched final-test languages.
- **H∞ target:** fixed `lambda=1.5`, exactly matching A-LQR. The generic optional
  lambda-sweep implementation remains available but is disabled for every run.
- **H∞ grid:** `R=1`, `Q/R in {0.01, 0.1, 1, 10}`, and
  `Qf/R in {0.01, 0.1, 0.316...}` at the fixed target.
- **Objective:** mean of the per-response score `0.9 × exact-answer accuracy +
  0.1 × (AXBench Overall / 2)`. AXBench Overall is itself the harmonic mean of
  Spanish adherence, instruction relevance, and fluency. A zero on either axis
  removes only that axis's contribution rather than collapsing the entire score.

## 5. Final evaluation

- Use one fixed aligned subset of 100 problem identities.
- Evaluate the same identities in Chinese, French, Japanese, Swahili, and Telugu.
- English and Spanish build the steering direction and are not primary test
  conditions.
- No controller is refit by language.

## Scoring

- **Accuracy:** exact final-number match with locale-aware comma/period decimal
  and thousands-separator parsing.
- **Spanish adherence:** deterministic AXBench rule score, 0 or 2.
- **Instruction relevance:** AXBench 0–2 rubric.
- **Fluency:** AXBench 0–2 rubric.
- **Overall steering:** per-response harmonic mean of the three steering scores.

## Evaluation size

- Active protocol per model: `100 × 5 languages × 3 methods = 1,500`
  generations.
- Historical Qwen and Llama S-PID generations remain archived but are outside
  the active comparison.
- H∞ selection per model: `12 × 100 = 1,200` cost-grid generations. There is no
  setpoint-multiplier sweep.
- There are no repeated final-evaluation seeds.

## References

- [MGSM paper](https://arxiv.org/abs/2210.03057)
- [MGSM dataset](https://huggingface.co/datasets/juletxara/mgsm)
- [AXBench paper](https://arxiv.org/abs/2501.17148)
- [Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Phi-4-mini-instruct model card](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [Granite-3.3-2B-Instruct model card](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)
