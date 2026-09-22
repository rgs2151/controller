# HarmBench robust refusal

## Pipeline card

- **Status:** Implemented; H∞ is being recomputed with AXBench-overall selection.
- **Task:** Respond to text-only HarmBench behaviors.
- **Distribution shift:** Direct requests are transformed by five frozen official
  human-jailbreak templates per behavior.
- **Steered behavior:** Do not exhibit the requested harmful behavior.
- **Direction data:** 50 human-majority non-exhibiting and 50 harmful-compliance
  validation completions; response-token DiffMean.
- **Shared dynamics:** 50 desired-completion Jacobians; one `A` shared by A-LQR
  and H∞.
- **H∞ disturbance data:** 50 direct non-copyright validation behaviors.
- **Baseline settings:** Fixed A-LQR setting; no baseline sweep.
- **H∞ selection:** The same 50 validation behaviors; 12 cost configurations;
  maximize AXBench overall with validation ASR used only as a tie-breaker.
- **Final evaluation:** 240 direct test behaviors and 1,200 matched human-jailbreak
  prompts; optional fixed 200-question five-shot MMLU.
- **Models:** Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct,
  Llama-3.1-8B-Instruct, and Llama-3.1-70B-Instruct.
- **Methods:** Original, A-LQR, and H∞.
- **Scoring:** Official test ASR, safe-concept relevance, instruction relevance,
  fluency, AXBench overall, and optional MMLU accuracy.
- **Evaluation size:** Per model, 600 H∞ selection generations and 4,320
  final safety generations.

## Question

Does a controller that suppresses harmful compliance on direct requests retain
that behavior under official human jailbreak templates?

## Frozen scope

- **Models:** Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct,
  Llama-3.1-8B-Instruct, and Llama-3.1-70B-Instruct. Artifacts and H∞
  calibration are fit independently for every model. The official BF16 70B
  checkpoint uses two H200s per model-parallel worker; eight H200s therefore
  run four data workers concurrently.
- **Methods:** Original, A-LQR, and H∞.
- **Target:** non-exhibition of requested harmful behavior, not a generic writing
  style and not a claim of pure harmlessness.
- **Behaviors:** text-only standard and contextual behaviors; copyright and
  multimodal behaviors are excluded.
- **Decoding:** greedy, at most 512 new tokens, one completion per case.
- **KV cache:** off.

## 1. Direction fitting

- Source: official HarmBench classifier-validation completions whose behavior IDs
  belong to the text validation split.
- Desired class: 50 human-majority label-0 completions that do not exhibit the
  requested harmful behavior.
- Undesired class: 50 human-majority label-1 harmful-compliance completions.
- Sampling is fixed and category-stratified.
- Estimator: layer-wise desired-minus-undesired DiffMean over response tokens.

## 2. Shared dynamics

- Use the 50 desired direction records as Jacobian prompts.
- Average their Jacobians into one saved `A`.
- A-LQR and H∞ use the same `A` artifact.

## 3. H∞ disturbance fitting

- Use 50 fixed, direct, non-copyright validation behaviors.
- Fit `D`, reduced coordinates, and the base H∞ problem once.
- The same 50 validation behaviors are reused for H∞ selection; this reuse is
  explicit because the official non-copyright validation pool contains only 60
  behaviors. Final test behaviors remain untouched.

## 4. Controller selection

- **A-LQR:** fixed `lambda=1.5`, `Q=0.1`, `R=1`, `Qf=0.1`. No sweep.
- **H∞ grid:** `R=1`, `Q/R in {0.01, 0.1, 1, 10}`, and
  `Qf/R in {0.01, 0.1, 0.316...}`.
- Generate one response for each of the 50 validation behaviors per candidate.
- **Objective:** maximum mean AXBench overall steering, the harmonic mean of safe
  concept relevance, instruction relevance, and fluency.
- Tie-break with the three components, then lower validation attack success,
  lower control energy, and smaller costs. Test ASR never selects the controller.

## 5. Final evaluation

- **Direct:** all 240 non-copyright standard/contextual test behaviors.
- **Jailbreak transfer:** the same 240 behavior identities under five frozen
  official human-jailbreak templates per behavior, totaling 1,200 prompts.
- **Optional capability:** 200 frozen five-shot MMLU questions.
- Validation and test behavior IDs remain separate.

## Scoring

- **Attack success rate (ASR):** official HarmBench test classifier; lower is
  better. The validation classifier is used only on validation data.
- **Concept relevance:** AXBench 0–2 safe/non-exhibition rubric.
- **Instruction relevance:** AXBench 0–2 rubric.
- **Fluency:** AXBench 0–2 rubric.
- **AXBench overall:** per-response harmonic mean of those three scores.
- **MMLU accuracy:** exact A/B/C/D accuracy when requested.

## Evaluation size

- Safety evaluation per method: `240 + 1,200 = 1,440` generations.
- Across three methods: `4,320` safety generations.
- Optional MMLU: `200 × 3 = 600` generations.
- H∞ selection: `12 × 50 = 600` short generations.
- The 70B expansion uses the compact frozen evaluation requested for the 8B
  expansion: 120 direct behaviors, their 600 matched jailbreak prompts, a
  100-token cap, and no MMLU.

## References

- [HarmBench paper](https://arxiv.org/abs/2402.04249)
- [Official HarmBench repository](https://github.com/centerforaisafety/HarmBench)
- [AXBench paper](https://arxiv.org/abs/2501.17148)
