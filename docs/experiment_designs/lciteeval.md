# L-CiteEval long-context steering

## Frozen experiment

Test whether feedback control preserves a fixed steering concept as irrelevant context grows, while retaining the model's ability to answer from the context and cite its evidence.

This experiment uses one controlled slice rather than the full L-CiteEval suite:

- **Dataset:** L-CiteEval-Length, HotpotQA.
- **Lengths:** approximately 8K, 16K, and 32K tokens.
- **Matched cases:** 40 underlying questions, each represented at all three lengths.
- **Models:** Qwen2.5-3B-Instruct and Llama-3.1-8B-Instruct.
- **Methods:** Original, S-PID, A-LQR, and H∞.
- **Decoding:** one deterministic generation per case; no repeated seeds.
- **Steering concept:** `positive sentiments and descriptions of enjoyable experiences`, concept 499 in the released AXBench Gemma-2-9B layer-20 data.

The complete evaluation is therefore `40 questions × 3 lengths × 4 methods × 2 models = 960` generations.
Each model–method–length result cell contains the same 40 question identities. The
three length columns therefore compare 40 answers at approximately 8K, the same
40 questions at approximately 16K, and the same 40 questions at approximately
32K. The 120 rows are 40 matched questions under three context conditions, not
120 different questions per condition.

## What the task is

Each example contains a question and a long collection of numbered passages. Most passages are irrelevant. The model is instructed to:

1. answer the question using only the supplied passages;
2. keep the answer concise and factual;
3. end every answer sentence with one to three passage citations such as `[12][37]`; and
4. cite only the minimum set of passages needed to support that sentence.

Use the official HotpotQA instruction, passage formatting, chat template, and one-shot demonstration unchanged. **One-shot means one worked example inside the prompt; it does not mean multiple sampled generations.**

HotpotQA is the selected task because it has short outputs, exact automatic answer scoring, and matched length conditions. The L-CiteEval paper also reports that Llama-3.1-8B loses roughly 20 points on this task as context grows from 8K to 32K, so the slice has a documented length effect before steering is introduced.

## Dataset

- **Source:** `Jonaszky123/L-CiteEval`.
- **Pinned revision:** `c79c928529593f478e6573c969cf73d22f0cf0f9`.
- **Configuration/file:** `L-CiteEval-Length/hotpotqa.json`.
- **Rows:** 120: 40 rows below 8K, 40 rows from 8K to 16K, and 40 rows from 16K to 32K.
- **Pairing:** the same 40 questions and gold answers appear in each length condition; the irrelevant padding passages change.
- **Use:** all 120 rows are the untouched test set. They are not used to fit a direction, estimate dynamics, fit H∞ disturbance geometry, or choose hyperparameters.

The current Hugging Face release is the implementation source of truth. It differs from the earlier paper description: the paper describes four Length tasks with 200 rows each, while the current release contains five tasks and 570 rows. This experiment uses all 120 released HotpotQA rows, not a further subsample.

## Steering and controller setup

### Semantic direction

- Use the released AXBench concept `positive sentiments and descriptions of enjoyable experiences`.
- The authoritative source is `pyvene/axbench-concept500` at revision
  `ad8a5d60c4616b599c24dd6689f05f696ec610f3`, variant
  `prod_9b_l20_v1`, concept ID `499`, corresponding to Gemma-2-9B-it,
  residual-stream layer 20, GemmaScope feature `27776`.
- Do not resolve concept ID `499` from another Concept500 variant: AXBench concept
  IDs are variant-specific, and ID `499` denotes different concepts in the 2B
  and 9B-layer-31 releases.
- The pinned variant contains exactly 72 unique positive inputs and 72 unique
  positive responses for this concept. Use all 72 positive responses.
- Its shared negative pool contains 216 responses split evenly across `text`,
  `math`, and `code`. Following AXBench `prepare_df`, filter that pool to the
  positive concept's `text` genre and use all 72 resulting negative responses.
- The authoritative local parquet SHA-256 is
  `5fb3042cc484a3e194f1043aab047bc1cfbe4efe884131d0113c75e9d2da599a`.
- Extract the direction separately in each target model's representation space.
- Keep the AXBench direction data disjoint from every L-CiteEval row.

These 144 responses, not the H∞ selection prompts, fit the semantic direction.
Following AXBench `DiffMean`, format each input and response with the target
model's chat template, remove BOS/padding/suffix tokens, collect residual states
for every remaining valid token, and form the layer-wise contrast

\[
v_k = \mu_k^{\mathrm{desired}} - \mu_k^{\mathrm{undesired}},
\qquad
\hat v_k = v_k / \lVert v_k \rVert_2.
\]

Here each mean is over all valid tokens in its class, exactly as in AXBench's
`DiffMean.train`. AXBench fits one selected intervention layer; our controller
requires a trajectory, so the same estimator is repeated independently at each
controlled layer. The existing shared artifact pipeline then uses the normalized
layer-wise directions to construct the semantic setpoint and the common reduced
coordinates used by A-LQR and H∞. A positive-only mean is not used: subtracting
a genre-matched undesired mean removes the model's generic response offset and
isolates the target concept.

The controller construction distribution is therefore AXBench-style short instructions. The 8K L-CiteEval condition is the shortest **transfer** condition, not the distribution from which the semantic direction was learned.

This concept is retained because it is concrete, lexically and semantically
recognizable, applicable to arbitrary instructions, and visibly changes model
responses without prescribing a fixed answer format. AXBench itself uses this
exact concept for its Appendix-N steering examples, including examples showing
successful injection and the transition from insufficient steering to excessive,
instruction-damaging steering. It is therefore a defensible, behaviorally active
concept rather than an invented project-specific target. We do not claim that it
is AXBench's single strongest concept: AXBench does not publish a per-concept
ranking establishing that claim, and transfer strength in Qwen and Llama must be
observed in this experiment rather than assumed.

### Shared artifacts

- Build the semantic setpoint and nominal dynamics once per model.
- Use the frozen controller artifact pipeline without changing the H∞ state, residual, disturbance, Riccati, or gain definitions.
- A-LQR and H∞ must use the same saved nominal matrix `A`.
- H∞ may compute `A` only when the shared A-LQR artifact does not exist; the calculation is the same shared Jacobian procedure.
- No controller is refit at 8K, 16K, or 32K.

### Fixed baseline settings

Do not sweep S-PID or A-LQR. Use the fixed concept-steering settings preserved in the upstream A-LQR Concepts code:

- **S-PID:** setpoint multiplier `1.5`, `Kp=0.5`, `Ki=0.5`, `Kd=0.01`.
- **A-LQR:** setpoint multiplier `1.5`, `Q=0.1 I`, `R=1 I`, `Qf=0.1 I`.

### Method composition

The first run uses Original, S-PID, A-LQR, and H∞, but these are configuration,
not a fixed four-way implementation. Each method writes an independent cache
namespace under the same model, dataset, length, calibration ID, and KV-cache
condition. Adding ODESteer later requires one ODESteer method adapter plus its
method key in this benchmark's TOML composition. It does not change L-CiteEval
loading, prompt construction, generation records, scorers, or any completed
method cache.

Every method adapter owns only its method-specific fit/calibration and the hook
or policy installed during generation. It receives the same prepared examples,
model revision, decoding configuration, and score stage as every other method.
Original is an ordinary registered method whose policy is empty, rather than a
separate evaluation path.

### H∞ selection

H∞ is the only swept method. Select it on 50 fixed, short AXBench-style
calibration instructions:

- Source the instructions from `tatsu-lab/alpaca_eval` at revision
  `2edc6fad8be6b14ea7230aabfd08188da6b8b814`, which contains 805 instructions.
- Mirror AXBench's concept-specific sampling rule: select 50 rows without
  replacement using pandas `sample(n=50, random_state=499)` on the pinned file.
- The source `alpaca_eval.json` SHA-256 is
  `d92b92c51e8f1962a21193abe74e6f727c2bc8286035f4041505ff38a7c3ae51`.
- The canonical JSON serialization of the 50 selected source indices and
  instructions has SHA-256
  `2991b504931d0a2f74539b0db43a775366b4c4d2dc6773b59b52ba380258ed5d`.
- The selection contains exactly 50 unique instructions and has zero exact-prompt
  overlap with the 72 positive and 72 genre-matched negative direction examples.
- Use the same 50 prompt identities for every H∞ candidate and both target
  models. Each prompt is generated once per candidate; there are no repeated
  seeds.
- These 50 prompts do not fit the concept direction. They exercise each already
  synthesized H∞ candidate so the AXBench scorers can select its control-cost
  configuration.
- Format the 200 disturbance instructions and all candidate-generation inputs
  with the same pinned target-model chat template used at runtime.

Sweep:

- `Q/R ∈ {0.01, 0.1, 1, 10}`;
- `Qf/R ∈ {0.01, 0.1, 0.316227766}`;
- `R=1`;
- no setpoint-multiplier sweep; use `1.5`;
- generate at most 128 new tokens per short calibration instruction, matching
  AXBench's steering evaluation length;

Score calibration generations only with the three independent AXBench OpenAI
judges: concept presence, instruction relevance, and fluency. Each returns an
integer in `{0,1,2}`. For each response, compute the AXBench overall score as the
harmonic mean of those three component scores, or zero if any component is zero.
The candidate score is the arithmetic mean of those 50 per-response overall
scores. Select the candidate with the highest candidate score; break exact ties
by higher mean instruction relevance and then higher mean fluency.

Do not run L-CiteEval answer-overlap or citation-NLI scoring during calibration.
Those metrics are defined only for the untouched L-CiteEval test rows and remain
evaluation outcomes rather than H∞ selection criteria. Save all three judge
scores, their explanations, the per-response harmonic means, and the selected
candidate record.

After selection, freeze the complete H∞ artifact and apply it unchanged at all three L-CiteEval lengths.

## Models and context validation

| Model | Pinned revision | Context used here | L-CiteEval precedent |
|---|---|---:|---|
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | 128K with static YaRN, factor 4, original maximum 32,768 | Evaluated in the L-CiteEval paper, including the Length benchmark |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | Native 128K | Evaluated in the L-CiteEval paper, including the Length benchmark |

Use the same Qwen YaRN configuration at every length so that the 8K-to-32K comparison does not also change the model configuration. Gemma-2-2B and the original Llama-3-8B are excluded because their native 8K windows cannot support the 16K and 32K conditions. Qwen-2.5-14B is excluded from this first experiment because the L-CiteEval paper did not report that exact checkpoint and it adds compute without improving direct comparability.

Before generation, tokenize the fully assembled prompt, including the one-shot demonstration and chat template. Assert that `input tokens + 200 generated tokens` fits the configured model window. Never truncate a passage or silently drop an example.

## Generation

- Use the official one-shot HotpotQA prompt.
- Maximum new tokens: 200.
- Temperature: 0.
- Top-p: 1.
- Sampling: off.
- Stop at the first newline or model end token, matching the official inference code.
- Evaluated-model KV cache: off for Original and all controllers, following the project-wide controlled-decoding decision.
- Save the prompt identifier, exact tokenized input length, generated answer, parsed citations, model revision, method configuration, device, start time, end time, and peak GPU memory.

All methods receive the same tokenized prompts in the same order. Original uses the same model-loading and decoding path with the intervention disabled.

## Scoring

Every generated answer is scored independently after generation. Scoring never triggers regeneration.

There is no single L-CiteEval judge. The score stage is the following explicit
set of independently selectable scorers:

| Scorer | Backend | Input | Output | Sends long context to an API? |
|---|---|---|---|---|
| `lcite_answer_overlap` | Deterministic Python | generated answer, released gold answer(s) | answer precision, recall, F1 in `[0,1]` | No |
| `lcite_citation_nli` | Local pinned `tasksource/deberta-base-long-nli` | each generated claim and only its cited passages | citation precision, recall, F1 in `[0,1]` | No |
| `axbench_concept_relevance` | OpenAI `gpt-4o-mini-2024-07-18` | concept description, generated answer | AXBench integer score `0–2` plus explanation | No |
| `axbench_instruction_relevance` | OpenAI `gpt-4o-mini-2024-07-18` | HotpotQA instruction/question, generated answer | AXBench integer score `0–2` plus explanation | No |
| `axbench_fluency` | OpenAI `gpt-4o-mini-2024-07-18` | generated answer | AXBench integer score `0–2` plus explanation | No |
| `axbench_overall` | Deterministic post-processing | the three saved AXBench component scores | harmonic mean in `[0,2]`, or zero if any component is zero | No |

The answer scorer is not model-based. The citation scorer is learned but runs
locally at pinned revision `04dcf11f844b07bc57015169fca2b7d6df8299d5`.
Only the three AXBench component scorers call an API, and their saved scores can
be recomputed or replaced without regenerating a model answer.

### L-CiteEval answer quality

- Strip citation markers from the answer.
- Apply the official normalized token overlap scorer against the gold answers.
- Report **answer recall** as the primary generation-quality measure, matching the L-CiteEval Length paper.
- Retain answer precision and answer F1 as diagnostics.
- This scorer is deterministic and requires no judge model.

### L-CiteEval citation quality

- Report citation precision, citation recall, and **citation F1**.
- Use the benchmark's `tasksource/deberta-base-long-nli` entailment procedure.
- Give the NLI model only each generated claim and the passages cited for that claim. It does not receive the full 8K–32K context.
- Citation F1 is the primary citation measure, matching the L-CiteEval Length paper.

### AXBench steering quality

Apply the three official AXBench 0–2 rubrics as separate judge calls:

- **Concept score:** whether the target positive-sentiment concept is naturally present in the answer.
- **Instruction score:** whether the answer addresses the HotpotQA question.
- **Fluency score:** whether the answer is readable and natural.

Use `gpt-4o-mini-2024-07-18`. Preserve the three AXBench rubrics and 0–2 scales, but return each batch through the pipeline's strict JSON schema rather than parsing the original textual `Rating: [[0|1|2]]` wrapper. This changes only the response transport, not the scoring criteria. Do not combine the three rubrics into one prompt; AXBench reports that unified judging can ignore a missing concept.

The judge inputs are deliberately short:

- concept judge: concept description plus generated answer;
- instruction judge: benchmark instruction, question, and generated answer;
- fluency judge: generated answer only.

**Never send the long passages to the OpenAI judge.** Correctness is already measured against the gold answer, and citation support is checked locally against only the cited passages.

For each response, compute the AXBench overall steering score as the harmonic mean of concept, instruction, and fluency scores, with an overall score of zero if any component is zero.

## Primary analysis

For each model and method, report the mean at 8K, 16K, and 32K for:

1. AXBench concept score;
2. AXBench overall steering score;
3. L-CiteEval answer recall; and
4. L-CiteEval citation F1.

The primary comparison is H∞ versus A-LQR on the same 40 question families:

- difference at 32K; and
- change from 8K to 32K.

Use a paired bootstrap over the 40 question families to give 95% confidence intervals for those method differences. Bootstrap resampling is post-processing; it does not require new model generations.

The main table is organized as three context-length columns. Each cell summarizes
40 responses, and the row alignment is by the same underlying question identity
across 8K, 16K, and 32K.

The result supports the intended claim only if H∞ preserves more steering at 32K without a corresponding collapse in answer recall or citation F1. Do not merge steering and L-CiteEval task quality into one paper metric; show the trade-off directly.

## Compute and API cost

- **Long-context generation:** 960 total generations.
- **Dataset context volume:** approximately 17.3 million context tokens before the fixed one-shot demonstration and chat-template overhead.
- **Repeated decoding:** none.
- **H∞ sweep:** 12 candidates × 50 fixed short instructions × 2 models = 1,200 short-context calibration generations, with no repeated seeds.
- **H∞ calibration judging:** 1,200 responses × 3 AXBench judges = 3,600 short OpenAI calls. No L-CiteEval answer or citation scorer runs during calibration.
- **OpenAI judging:** 960 responses × 3 independent AXBench rubrics = 2,880 short calls.

At the current GPT-4o-mini rates, the AXBench judging should remain comfortably below one US dollar because no long context is transmitted. The exact cost must be computed from saved API token usage. The native answer and citation scorers incur no OpenAI cost. Long-context controlled generation, not judging, is the dominant expense.

## Required preflight

Before the full run, perform one smoke case per model at each length for all four methods and verify:

- NLTK 3.9.1 and its `punkt_tab` sentence-tokenizer resource are installed for the released citation scorer;
- the fully assembled prompt fits without truncation;
- citations survive parsing;
- controller hooks run with KV cache off;
- Original and controlled generations use identical decoding settings;
- all six native metric fields and four AXBench score fields are written;
- the 40 question identities align across the three length conditions; and
- H∞ writes the existing diagnostic bundle unchanged.

The smoke outputs are disposable and are not part of the reported 960 generations.

## References

- [L-CiteEval paper](https://arxiv.org/abs/2410.02115)
- [L-CiteEval dataset](https://huggingface.co/datasets/Jonaszky123/L-CiteEval)
- [L-CiteEval code](https://github.com/LCM-Lab/L-CITEEVAL)
- [AXBench paper](https://arxiv.org/abs/2501.17148)
- [AXBench code](https://github.com/stanfordnlp/axbench)
- [AXBench Concept500 release](https://huggingface.co/datasets/pyvene/axbench-concept500)
- [AlpacaEval instruction source](https://huggingface.co/datasets/tatsu-lab/alpaca_eval)
- [Qwen2.5-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Llama-3.1-8B-Instruct model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
