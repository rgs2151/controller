# L-CiteEval Small fast iteration

## Pipeline card

- **Status:** Implemented as an independent benchmark; not yet executed.
- **Task:** Answer ten matched HotpotQA questions from long documents and cite
  supporting passages.
- **Distribution shift:** 8K is the controller-selection and in-distribution
  condition; the same ten question identities at 16K and 32K are available as
  out-of-distribution conditions, with only 32K enabled in the default run.
- **Steered behavior:** Positive sentiments and descriptions of enjoyable
  experiences, using AXBench concept 499.
- **Direction data:** 72 positive and 72 genre-matched negative AXBench examples.
- **Shared dynamics:** 50 desired AXBench prompt Jacobians; one saved `A` is
  shared by A-LQR and H∞.
- **H∞ disturbance data:** 200 frozen AlpacaEval prompts.
- **Baseline settings:** A-LQR uses fixed `lambda=1.5`, `Q=0.1`, `R=1`, and
  `Qf=0.1`; it is not swept.
- **H∞ selection:** Twelve `Q/R,Qf/R` configurations on ten fixed 8K questions;
  maximize `0.40 answer recall + 0.40 citation F1 + 0.10 normalized fluency +
  0.10 normalized concept relevance`.
- **Final evaluation:** Ten matched questions at 8K and 32K by default, with 16K
  retained as an optional configured condition; deterministic generation, KV
  cache off, and at most 128 new tokens.
- **Model:** Llama-3.2-1B-Instruct.
- **Methods:** Original, A-LQR, and H∞.
- **Scoring:** L-CiteEval answer overlap and AutoAIS citation metrics plus
  AXBench concept relevance, instruction relevance, fluency, and overall steering.
- **Evaluation size:** 120 H∞ selection generations and 60 default final
  generations; enabling 16K adds 30 final generations.

The 8K H∞ values are development results because those ten prompts select the
controller. The 16K and 32K results measure context-length transfer. This unit
does not read or overwrite either existing L-CiteEval benchmark cache.
