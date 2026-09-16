# L-CiteEval long-context language transfer

## Pipeline card

- **Status:** Spanish-transfer pipeline implemented; Llama validation/run is next,
  the matching Qwen rerun is deferred, and the historical Qwen sentiment run is separate.
- **Task:** Answer HotpotQA questions from long evidence documents and cite the
  supporting passages.
- **Distribution shift:** The same 40 questions are presented at approximately
  8K and 16K context lengths; 32K is deferred.
- **Steered behavior:** Answer only in Spanish without an explicit Spanish
  instruction in the evaluation prompt.
- **Direction data:** 250 matched MGSM English/Spanish question pairs; paired
  Spanish-minus-English DiffMean fitted separately in each model.
- **Shared dynamics:** 50 frozen Spanish MGSM question Jacobians; one `A` per
  model shared by A-LQR and H∞.
- **H∞ disturbance data:** 200 upstream 2WikiMultihopQA training questions
  formatted as approximately 8K L-Cite citation prompts.
- **Baseline settings:** Fixed S-PID and A-LQR settings; no baseline sweep.
- **H∞ selection:** 40 official L-CiteEval 2Wiki base-context questions; 12 cost
  configurations; maximize the Spanish-adherence, instruction-relevance, and
  fluency harmonic mean.
- **Final evaluation:** 40 matched HotpotQA questions × 2 lengths; deterministic
  generation with a 200-token cap.
- **Models:** Qwen2.5-3B-Instruct and Llama-3.1-8B-Instruct; Llama runs next and
  the comparable Qwen Spanish run comes later.
- **Methods:** Original, S-PID, A-LQR, and H∞.
- **Scoring:** Bilingual answer correctness, bilingual citation
  recall/precision/F1, Spanish adherence, instruction relevance, fluency, and
  overall steering.
- **Evaluation size:** Per model, 480 H∞ selection generations and 320 final
  generations.

## Status

This document separates two experiments that must not be pooled:

- **Completed historical run:** Qwen2.5-3B-Instruct, steered toward positive
  sentiment with AXBench concept 499. Its 8K/16K results remain frozen.
- **Next run:** Llama-3.1-8B-Instruct, steered to answer in Spanish. The isolated
  `lciteeval_spanish` implementation is ready for validation on Local 2.

Because the model and steering target both change, the two runs are separate
ablations. They are not a model-scaling comparison.

## Question

Can feedback control preserve Spanish output as the English evidence context
grows from approximately 8K to 16K tokens, without losing answer or citation
quality?

## Frozen scope for the next run

- **Model:** Llama-3.1-8B-Instruct.
- **Methods:** Original, S-PID, A-LQR, and H∞.
- **Final task:** L-CiteEval-Length HotpotQA.
- **Lengths:** 8K and 16K. The matched 32K condition is deferred and remains
  independently composable.
- **Steering rule:** answer only in Spanish; the evaluation prompt itself does
  not request Spanish.
- **Decoding:** deterministic, one generation per question, at most 200 new
  tokens.
- **KV cache:** off.

## 1. Direction fitting

- Reuse the frozen MGSM direction **text corpus**: all 250 aligned English and
  Spanish question pairs.
- Undesired class: each English question.
- Desired class: its matched Spanish translation.
- Use the same chat formatting and paired DiffMean procedure as MGSM.
- Recompute the residual direction and setpoint in Llama-3.1-8B's own
  representation space. Do not copy a Qwen direction tensor.
- This stage defines the language axis only. It does not fit long-context
  disturbance geometry and does not select a controller.

## 2. Shared dynamics

- Use the same frozen 50 Spanish MGSM direction prompts used by the MGSM
  artifact procedure.
- Average their prompt Jacobians into one nominal matrix `A` for Llama-3.1-8B.
- A-LQR and H∞ use this exact saved `A`.
- Do not reuse the Qwen MGSM `A`; dynamics are model-specific.

## 3. H∞ disturbance fitting

- Source: 200 unique questions from the upstream 2WikiMultihopQA **training**
  split, outside every official L-CiteEval evaluation row.
- Format each question as an L-Cite-style multi-document QA prompt with numbered
  passages and citation instructions.
- Use the short/base context construction, near the 8K condition.
- Fit the reduced coordinates and disturbance channels `D` from these prompts.
- Save the split IDs and formatted prompts so this stage is exactly repeatable.

Why 2WikiMultihopQA: it matches the final task's multi-hop, multi-document,
citation structure while remaining separate from the final HotpotQA questions.
GSM8K or generic short instructions are not used for this stage.

## 4. Controller selection

- **S-PID:** fixed `lambda=1.5`, `Kp=0.5`, `Ki=0.5`, `Kd=0.01`.
- **A-LQR:** fixed once at `lambda=1.5`, `Q=0.1`, `R=1`, `Qf=0.1`; no sweep.
- **H∞ development set:** the 40 unique shortest/base-context question families
  from the official L-CiteEval 2WikiMultihopQA release.
- These 40 questions are disjoint from the upstream 2Wiki training questions
  used for disturbance fitting and from the final HotpotQA questions.
- **H∞ grid:** `R=1`, `Q/R in {0.01, 0.1, 1, 10}`, and
  `Qf/R in {0.01, 0.1, 0.316...}`.
- One generation per question per candidate: `12 × 40 = 480` selection
  generations. “40 prompts” means 40 independent generations, not a 40-shot
  prompt.
- **Objective:** maximum harmonic mean of Spanish adherence, instruction
  relevance, and fluency.
- L-Cite answer and citation metrics are diagnostics during selection, not the
  optimization objective.

## 5. Final evaluation

- Use the same 40 HotpotQA question identities at approximately 8K and 16K.
- The 8K and 16K rows differ in padding/context length, not in the underlying
  question or answer.
- Every method receives the same prompts and decoding configuration.
- No direction, `A`, `D`, setpoint, gain, or hyperparameter is refit by length.
- Per model: `40 questions × 2 lengths × 4 methods = 320` generations.

## Scoring

### Task quality

- **Answer correctness:** a bilingual OpenAI judge compares the raw Spanish
  response directly with the English question and official answer. Scores are
  0 (incorrect/absent), 0.5 (partially correct), or 1 (fully correct).
- **Citation recall, precision, and F1:** a bilingual OpenAI judge compares each
  raw Spanish claim directly with its cited English passages. Code computes the
  three metrics from the judge's claim-support and citation-necessity decisions.
- The response is never translated or language-normalized. Citation markers and
  the stored Spanish generation remain untouched.

### Steering quality

- **Spanish adherence:** deterministic AXBench rule score, 0 or 2, computed on
  the raw response.
- **Instruction relevance:** AXBench 0–2 rubric on the raw response.
- **Fluency:** AXBench 0–2 rubric on the raw response.
- **Overall steering:** per-response harmonic mean of those three scores.

Task quality and steering quality are reported side by side. They are not merged
into one paper metric.

## Data separation

| Role | Dataset | Count |
|---|---|---:|
| Direction | Matched MGSM English/Spanish questions | 250 pairs |
| Shared `A` | Spanish side of the frozen MGSM pairs | 50 prompts |
| H∞ disturbance fit | Upstream 2WikiMultihopQA train | 200 questions |
| H∞ selection | Official L-CiteEval 2Wiki shortest/base rows | 40 questions |
| Final evaluation | Official L-CiteEval-Length HotpotQA | 40 questions × 2 lengths |

The official L-CiteEval 2Wiki release contains only 40 unique question families,
so it cannot honestly supply both 200 disturbance prompts and a disjoint
selection set. The 200 disturbance prompts therefore come from upstream 2Wiki
training data; the official 40 are reserved for selection.

## References

- [L-CiteEval paper](https://arxiv.org/abs/2410.02115)
- [L-CiteEval dataset](https://huggingface.co/datasets/Jonaszky123/L-CiteEval)
- [L-CiteEval code](https://github.com/LCM-Lab/L-CITEEVAL)
- [2WikiMultihopQA paper](https://arxiv.org/abs/2011.01060)
- [MGSM dataset](https://huggingface.co/datasets/juletxara/mgsm)
- [AXBench paper](https://arxiv.org/abs/2501.17148)
