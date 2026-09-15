# MGSM multilingual language-steering transfer

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-15
- Verification Status: LOCAL END-TO-END SMOKE VERIFIED (2026-09-15)
- Version Label: code_plan_v1

## Draft experiment

Test whether a controller that learns an English-to-Spanish language direction can
preserve that steering objective when the same mathematical problem is presented
in languages never used to construct or select the controller, without destroying
the model's ability to solve the problem.

- **Evaluation dataset:** MGSM.
- **Construction language:** English.
- **Desired response language:** Spanish.
- **Transfer variable:** language of the input problem.
- **Models:** Qwen3-4B and Qwen3-8B.
- **Methods:** Original, S-PID, A-LQR, and H∞.
- **Decoding:** one deterministic generation per problem; no repeated seeds.
- **Primary outcomes:** exact numerical answer accuracy and Spanish rule following.

The main hypothesis is that H∞ retains Spanish response steering more consistently
than A-LQR as the input language changes, while preserving comparable mathematical
accuracy.

## What MGSM contains

The Hugging Face release contains the same 250 GSM8K test problems in 11 language
configurations:

- English (`en`);
- Spanish (`es`);
- Bengali (`bn`);
- Chinese (`zh`);
- French (`fr`);
- German (`de`);
- Japanese (`ja`);
- Russian (`ru`);
- Swahili (`sw`);
- Telugu (`te`); and
- Thai (`th`).

The paper describes ten translations **in addition to English**. Consequently,
after English and Spanish there are nine transfer languages, not eight. Each
configuration has eight translated few-shot exemplars and the same 250 matched
test problems.

- **Source:** `juletxara/mgsm`.
- **Pinned revision:** `b2f13d426afe3be8d69a7e739b36724db8b66bbc`.
- **Train split:** 8 native-language few-shot exemplars per language, including
  translated step-by-step answers and equation solutions.
- **Test split:** 250 matched problems per language, containing translated
  questions and numeric gold answers but no gold reasoning traces.
- **Direction use:** all 250 matched English/Spanish problem pairs.
- **Transfer-test use:** the corresponding 250 problems in the other nine
  languages.

Because the direction uses every English/Spanish pair, English and Spanish are
construction diagnostics rather than held-out evaluation conditions. The primary
held-out comparison consists of the nine other language versions. This is a
transductive language-transfer design: problem identities are shared across
languages, while the input languages used for the reported transfer comparison are
not used to construct the direction.

## What the original MGSM paper measured

MGSM evaluates arithmetic reasoning with **exact-match accuracy on the final
numeric answer**. The gold answer is an Arabic numeral in every language. The
paper reports accuracy by language and macro-averaged across languages while
comparing direct, native-language chain-of-thought, English chain-of-thought, and
translation-based prompting conditions. It does not assign partial credit to a
correct reasoning trace with an incorrect final number.

This experiment uses the paper's native-language eight-shot chain-of-thought
condition because a reasoning trace supplies enough natural language to measure
whether the response has shifted into Spanish. One deterministic completion is
used instead of self-consistency or repeated sampling.

## Steering direction

Use the actual AXBench rule:

> `respond only in Spanish, and no other language is allowed`

Construct the direction from all 250 matched English/Spanish pairs in MGSM:

1. Align all 250 English and Spanish rows by problem identity.
2. Use the English question text as the undesired example.
3. Use the matched Spanish translation of that question as the desired example.
4. Apply the same minimal chat wrapper to both languages; do not add a worked
   solution or output-language instruction.
5. Verify that each pair has the same `answer_number` before artifact construction.
6. Materialize the 250 aligned question pairs once and reuse them for every method
   and model.

The MGSM test rows contain translated questions and numeric gold answers, not
translated worked solutions. Using matched question pairs is therefore the direct
way to use the complete released English/Spanish set. Keeping problem identity and
mathematical content matched means the contrast primarily captures language rather
than problem identity or reasoning content.

Following AXBench `DiffMean`, format each question with the target model's own chat
template, remove BOS, padding, and suffix tokens, collect the valid-token residual
states, and compute independently at every controlled layer

\[
v_k = \mu_k^{\mathrm{Spanish}}-\mu_k^{\mathrm{English}},
\qquad
\hat v_k = v_k / \lVert v_k \rVert_2.
\]

The paired text corpus is shared, but the residual direction is extracted
separately in each target model's representation space.

## Controller artifacts

- Build the semantic setpoint and nominal dynamics once per model from the frozen
  direction corpus and the project's artifact prompts.
- Use the frozen artifact pipeline without changing the H∞ state, residual,
  disturbance, Riccati, or gain definitions.
- A-LQR and H∞ use the same saved nominal matrix `A`.
- H∞ may compute `A` only when the shared A-LQR artifact is absent, using the
  same shared Jacobian procedure.
- No controller is refit for an MGSM language.

### Fixed non-H∞ methods

Do not sweep S-PID or A-LQR. Use the source-preserved concept-steering settings:

- **S-PID:** setpoint multiplier `1.5`, `Kp=0.5`, `Ki=0.5`, `Kd=0.01`.
- **A-LQR:** setpoint multiplier `1.5`, `Q=0.1 I`, `R=1 I`, `Qf=0.1 I`.

### H∞ selection

H∞ is the only swept method. Use 50 fixed GSM8K training problems that are not
part of MGSM. Present those calibration problems in English and use the same 50
identities for every H∞ candidate and model. Save every calibration generation,
component score, candidate mean, and the selected configuration.

Sweep:

- `Q/R ∈ {0.01, 0.1, 1, 10}`;
- `Qf/R ∈ {0.01, 0.1, 0.316227766}`;
- `R=1`;
- setpoint multiplier `1.5`; and
- no setpoint, lambda, or attenuation-multiplier sweep.

Score each calibration response with the three AXBench components:

1. Spanish rule following;
2. instruction relevance; and
3. fluency.

Each component is on the AXBench `0–2` scale. Compute the per-response harmonic
mean, assigning zero when any component is zero, and select the candidate with the
highest mean over the 50 responses. Break exact ties by higher instruction
relevance and then higher fluency. Do not use MGSM test accuracy to select the
controller.

## Evaluation conditions

For every one of the 250 matched problem identities, run the nine transfer-language
versions through every method. The same frozen controller is used throughout.
English and Spanish generations may be retained as construction diagnostics, but
they are not included in the held-out transfer claim.

- **English input:** direction-construction diagnostic.
- **Spanish input:** desired-language construction diagnostic.
- **Nine other input languages:** transfer conditions.

The prompt requests a worked solution and a final Arabic-numeral answer but does
not ask for Spanish. Spanish output should therefore be caused by the controller,
not by an explicit output-language instruction. Use the eight official
native-language MGSM exemplars for the corresponding input language.

Generation settings:

- maximum new tokens: 512;
- temperature: 0;
- top-p: 1;
- sampling: off;
- one generation per problem;
- evaluated-model KV cache: off for all methods; and
- Qwen3 thinking mode: off for both sizes, so they use one comparable visible
  reasoning-and-answer channel.

Save the problem identity, input language, fully assembled prompt, tokenized input
length, raw generation, extracted final number, model revision, method
configuration, controller artifact ID, device, start time, end time, and peak GPU
memory.

## Scoring

Generation and scoring are separate stages. Scoring never regenerates an answer.

| Scorer | Backend | Output |
|---|---|---|
| `mgsm_exact_match` | Deterministic Python | Final-number correctness in `{0,1}` |
| `axbench_rule_spanish` | AXBench deterministic rule evaluator | Spanish adherence in `{0,2}` |
| `axbench_instruction_relevance` | AXBench OpenAI rubric | Relevance in `{0,1,2}` plus explanation |
| `axbench_fluency` | AXBench OpenAI rubric | Fluency in `{0,1,2}` plus explanation |
| `mgsm_axbench_overall` | Deterministic post-processing | Harmonic mean in `[0,2]`, or zero if any component is zero |

### Mathematical accuracy

Extract the final Arabic-numeral answer with one language-independent parser and
compare it exactly with `answer_number`. Ignore comma separators and normalize a
trailing `.0`. A missing or unparseable answer is incorrect. Retain every raw
generation for audit.

The language-independent parser is necessary because successful steering can
change the final-answer prefix from the input language to Spanish. It preserves
the MGSM paper's substantive measure—exact final-number accuracy—without making
the parser itself dependent on whether steering succeeded.

### AXBench steering quality

For this rule-type concept, AXBench does not use its concept-relevance LM judge as
the first component. It uses its deterministic Spanish rule evaluator. The other
two components retain the official independent 0–2 instruction-relevance and
fluency rubrics.

- Spanish rule scorer input: generated response.
- Instruction scorer input: task instruction, current MGSM question, and response.
- Fluency scorer input: generated response.

Do not send the eight demonstrations to the OpenAI scorers. They are not needed
to judge whether the response addresses the current question or is fluent.

## Models

| Model | Pinned revision | Why it is suitable |
|---|---|---|
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | Primary model requested for the experiment. Qwen documents support for 119 languages and dialects, explicitly including every MGSM language. |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | Same documented 119-language coverage at a larger scale, giving a clean within-family comparison. |

Both models explicitly support every MGSM language. Keeping the experiment within
one model family avoids confounding the controller comparison with different chat
templates and multilingual training families.

Before artifact construction, run one smoke example in every language for each
model and verify readable tokenization, non-empty generation, numeric extraction,
and controller-hook compatibility. This is a support preflight, not reported
benchmark evidence.

## Primary analysis

For each model, method, and input language, report:

1. MGSM exact-match accuracy over the same 250 problem identities;
2. mean Spanish rule-following score;
3. mean AXBench instruction relevance;
4. mean AXBench fluency;
5. mean AXBench overall score.

The main figure has input language on the x-axis and three panels:

- **Math accuracy:** exact final-number accuracy.
- **Spanish steering:** mean Spanish rule-following score.
- **Overall steering:** mean AXBench harmonic-mean score.

Show all four methods. Treat English and Spanish as anchors and visually separate
the nine transfer languages. The main comparison is H∞ versus A-LQR on the same
problem identities.

Use paired bootstrap confidence intervals over the 250 problem identities for
H∞−A-LQR differences within each language. Also report the macro-average over
the nine transfer languages, weighting every language equally.

The intended claim is supported only if H∞ improves Spanish adherence or AXBench
overall steering across transfer languages without a disproportionate loss of
mathematical accuracy. Spanish adherence remains a 0–2 score, not a rate.

## Compute and API size

- **Held-out test generations per model:**
  `250 problems × 9 transfer languages × 4 methods = 9,000`.
- **Two-model held-out evaluation:** `18,000` test generations.
- **Optional English/Spanish construction diagnostics:**
  `250 problems × 2 languages × 4 methods × 2 models = 4,000` additional
  generations.
- **H∞ calibration:** `12 candidates × 50 prompts × 2 models = 1,200` short
  generations. This is separate from the 18,000 held-out evaluation generations.
- **Repeated test decoding:** none.
- **OpenAI scoring:** two short judge calls per generated response; Spanish rule
  following and mathematical accuracy are local deterministic scorers.

## Required preflight

Before the full run, verify:

- all 250 problem identities align exactly across the 11 language configurations;
- all 250 English/Spanish direction pairs align by problem identity and
  `answer_number`;
- the 50 H∞ selection prompts are outside MGSM;
- the official eight exemplars are loaded for every input language;
- the final-number parser is invariant to the language of the answer prefix;
- Qwen3 thinking mode and evaluated-model KV cache are recorded and fixed;
- A-LQR and H∞ reference the same saved `A` artifact;
- the three AXBench component scores and harmonic mean are written;
- H∞ writes its existing diagnostic bundle unchanged; and
- adding another model or method creates a new cache namespace without
  overwriting completed results.

## References

- [MGSM paper](https://arxiv.org/abs/2210.03057)
- [MGSM Hugging Face dataset](https://huggingface.co/datasets/juletxara/mgsm)
- [Official MGSM repository](https://github.com/google-research/url-nlp/tree/main/mgsm)
- [AXBench paper](https://arxiv.org/abs/2501.17148)
- [AXBench code](https://github.com/stanfordnlp/axbench)
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
- [Qwen3 language coverage](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
