# L-CiteEval Small fast iteration

## Pipeline card

- **Status:** Implemented as an independent benchmark; not yet executed.
- **Task:** Answer ten matched HotpotQA questions from long documents and cite
  supporting passages.
- **Distribution shift:** 8K is the controller-selection and in-distribution
  condition; the same ten question identities at 16K and 32K are available as
  out-of-distribution conditions, with only 32K enabled in the default run.
- **Steered behavior:** Respond only in Spanish, with no other language.
- **Direction data:** All 250 matched MGSM English/Spanish question pairs;
  Spanish-minus-English DiffMean is fit separately in each evaluated model.
- **Shared dynamics:** 50 frozen upstream 2WikiMultihopQA prompts expanded to
  approximately 8K; one saved `A` is shared by A-LQR and H∞.
- **H∞ disturbance data:** 200 disjoint frozen upstream 2WikiMultihopQA prompts
  expanded to approximately 8K.
- **Baseline settings:** A-LQR uses fixed `lambda=1.5`, `Q=0.1`, `R=1`, and
  `Qf=0.1`; it is not swept.
- **H∞ selection:** Twelve `Q/R,Qf/R` configurations on ten fixed official
  2WikiMultihopQA L-CiteEval prompts at approximately 8K. All four components
  use OpenAI judgments: maximize `0.40 bilingual semantic answer recall + 0.40
  bilingual citation F1 + 0.10 normalized Spanish concept relevance + 0.10
  normalized fluency`.
- **Prompt ceiling:** The released one-shot 8K prompts tokenize to about
  8.3--8.8K after the chat template, so artifact/calibration processing uses a
  10K ceiling to avoid truncating the official 8K condition.
- **Final evaluation:** Ten matched questions at 8K and 32K by default, with 16K
  retained as an optional configured condition; deterministic generation, KV
  cache off, and at most 128 new tokens.
- **Model:** Llama-3.2-1B-Instruct.
- **Methods:** Original, A-LQR, and H∞.
- **Scoring:** OpenAI bilingual semantic answer recall; citation F1 computed from
  OpenAI bilingual AutoAIS entailment decisions; OpenAI AXBench Spanish concept
  relevance; and OpenAI AXBench fluency.
- **Evaluation size:** 120 H∞ selection generations and 60 default final
  generations; enabling 16K adds 30 final generations.

The 8K H∞ values are development results because those ten prompts select the
controller. The 16K and 32K results measure context-length transfer. This unit
does not read or overwrite either existing L-CiteEval benchmark cache.
