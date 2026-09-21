# Experiment designs

These documents specify the scientific protocol. They are not run logs and they
do not infer missing choices from old caches.

## Pipeline card template

Every experiment-design file begins with this card. Whenever a pipeline is
proposed, changed, or launched, report this card before giving implementation
details.

- **Status:** What is completed, what runs next, and what is deferred.
- **Task:** The model's actual evaluation task.
- **Distribution shift:** What changes between the base and transfer conditions.
- **Steered behavior:** The exact behavior or rule being controlled.
- **Direction data:** Desired and undesired sources, counts, and estimator.
- **Shared dynamics:** Jacobian source/count and confirmation that A-LQR and H∞
  share one `A` per model.
- **H∞ disturbance data:** Dataset and count used to fit `D`.
- **Baseline settings:** Frozen S-PID/A-LQR settings; no hidden baseline sweep.
- **H∞ selection:** Development set, candidate grid, and selection score.
- **Final evaluation:** Untouched datasets, samples, repetitions, and decoding.
- **Models:** Every model belonging to the analysis.
- **Methods:** Every reported method.
- **Scoring:** Task and steering measures, kept distinct.
- **Evaluation size:** Calibration and final-generation counts.

Keep each item to one sentence. Model-specific math belongs under **Models** or
in the detailed design, not inside unrelated data-stage descriptions.

## The five stages

Every benchmark must state these stages separately:

1. **Direction fitting:** desired and undesired examples define the behavior axis
   and semantic setpoint.
2. **Shared dynamics:** prompt Jacobians are averaged into one nominal matrix
   `A` per model and benchmark. A-LQR and H∞ use that exact saved matrix.
3. **H∞ disturbance fitting:** a separate prompt set estimates the disturbance
   channels `D`. This step also synthesizes the candidate H∞ controllers.
4. **Controller selection:** a held-out development set chooses one H∞
   configuration. A-LQR and the other baselines use frozen settings and are not
   swept.
5. **Final evaluation:** untouched test prompts compare the frozen controllers.
   Scoring reads saved generations and cannot change controller selection.

Direction data, disturbance data, selection data, and final evaluation data are
different scientific roles even when they originate from the same dataset.

## Current inventory

| Benchmark | Steering target | Direction data | H∞ disturbance data | H∞ selection | Final evaluation |
|---|---|---|---|---|---|
| [Truthfulness](truthfulness.md) | Truthful, non-misleading answers | TruthfulQA true vs false answers | 200 disjoint TruthfulQA prompts | 50 disjoint TruthfulQA prompts | English and Spanish TruthfulQA; optional MMLU |
| [Toxicity](toxicity.md) | Non-toxic language | RTP non-toxic vs toxic prompts | 200 disjoint RTP prompts | 50 disjoint RTP prompts | RTP; optional MMLU |
| [MGSM](mgsm_language_transfer.md) | Answer only in Spanish | 250 matched Spanish vs English questions | 200 GSM8K-train prompts | 50 disjoint GSM8K-train prompts | 100 matched problems in five held-out languages |
| [L-CiteEval](lciteeval.md) | New run: answer in Spanish | 250 matched MGSM Spanish vs English questions | 200 upstream 2WikiMultihopQA training questions | 10 frozen official L-CiteEval 2Wiki questions | 40 matched HotpotQA questions at 8K and 16K |
| [L-CiteEval Small](lciteeval_small.md) | Positive sentiment and enjoyable experiences | 72 positive vs 72 matched negative AXBench examples | 200 frozen AlpacaEval prompts | 10 HotpotQA questions at 8K | The same 10 HotpotQA questions at 8K and 32K |
| [HarmBench](harmful.md) | Non-exhibition of harmful behavior | 50 safe/non-exhibiting vs 50 harmful-compliance completions | 50 direct validation behaviors | The same 50 validation behaviors | 240 direct and 1,200 human-jailbreak prompts; optional MMLU |

## Shared rules

- Artifacts are model-specific. Text records may be shared, but directions,
  setpoints, `A`, `D`, gains, and `gamma*` are recomputed for each model.
- H∞ may load the saved A-LQR `A`; if it is absent, it computes the same shared
  artifact. H∞ does not use a different Jacobian definition.
- Evaluated-model KV cache is off for reported runs. Cache-on is retained only as
  a separately named ablation.
- H∞ selection maximizes the per-response harmonic mean of concept relevance,
  instruction relevance, and fluency unless a design explicitly states another
  objective.
- Final task metrics remain separate from the steering-selection metric.
- Every split, seed, model revision, controller selection, start time, finish
  time, host, and GPU assignment is recorded with the run.
