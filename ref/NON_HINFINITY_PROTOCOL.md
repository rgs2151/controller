# Frozen non-H-infinity benchmark protocol

This audit covers Original, ITI, ActAdd, Mean-AcT, Linear-AcT, PID-AcT,
ODESteer, S-PID, and A-LQR. H-infinity is intentionally excluded.

## Final evaluation

- Toxicity: 5 × 1,000 RealToxicityPrompts generations.
- Truthfulness: 5 × all 817 TruthfulQA generation questions.
- Generation: temperature 1, top-p 0.3, repetition penalty 1.2; maximum 100
  new tokens for toxicity and 50 for truthfulness.
- Truthfulness scoring: `allenai/truthfulqa-truth-judge-llama2-7B` and
  `allenai/truthfulqa-info-judge-llama2-7B`, with the exact model-card rubrics.
- MMLU: 5-shot. Toxicity PPL: Mistral-7B. Toxicity classification:
  `s-nlp/roberta_toxicity_classifier`.
- Original and LFS-controlled RTP decoding use `use_cache=False`; Original MMLU
  uses `use_cache=True`, while LFS-controlled MMLU uses `use_cache=False`.

Fifty prompts was a pilot size and is not part of this protocol.

## Method audit

| Method | One-time fit | Final parameters | Source status |
|---|---|---|---|
| Original | none | none | frozen |
| A-LQR | 200/200 setpoint; 50 toxicity or 35 truthfulness Jacobians | Gemma-2-2B toxicity is λ=3.5, Q=.1, R=1, Qf=.1; truthfulness is λ=3, Q=.1, R=1, Qf=.3 | frozen for recorded rows |
| S-PID | same 200/200 setpoint; no Jacobians | fixed PID gains plus λ from the preserved source grid | explicit development selection still required where the grid has multiple λ values |
| ActAdd | 100/100 position-wise means | preserved model-specific layer and strength | frozen |
| ITI | 80/80, 80/20 stratified probe split, max length 50 | top-head count and α from preserved source grid | explicit development selection still required; final Gemma choice is not preserved |
| Mean-AcT | 200/200 toxicity or 400/400 truthfulness | first four matched modules, strength 1 | frozen |
| Linear-AcT | 200/200 toxicity or 400/400 truthfulness | first four matched modules, q_0_100 mask, strength 1 | frozen |
| PID-AcT | 200/200 toxicity or 400/400 truthfulness | first four matched modules, source 0.7 and 0.005 terms, strength 1 | frozen, including the source's scalar history reduction |
| ODESteer | 5000/5000 toxicity or 1800/1800 truthfulness | preserved model-specific layer/time; Euler, 10 steps, 8000 components, degree 2, γ=.1, c0=1 | frozen only where a final selection is preserved |

The final runner accepts exactly one selected configuration. It does not tune
on the final benchmark. ITI and S-PID cannot run a final row until an explicit
development-set selection from the preserved grids has been recorded; this
prevents an undocumented guess or an accidental full-evaluation sweep.

For toxicity, λ=3.5 is the strongest candidate in the paper-producing Gemma
evaluation script and is fixed according to the paper table's stated selection
rule: greatest toxicity decrease subject to acceptable PPL. Final evaluation
does not repeat the source sweep. The 1,000 five-shot MMLU questions are pinned
once and shared across methods; this removes the source script's avoidable
method-to-method question-set noise without changing the task or rubric.

## Source snapshots

- A-LQR implementation: `trustworthyrobotics/lqr-activation-steering`, commit
  `c2e0c8450797e0dc234b2c53475267a2cfb2457a`.
- A-LQR comparison adapters: the same repository, commit
  `11b49bad15c02f0a23d9481980d4b88f2d6313a5`, immediately before their removal.
- A-LQR paper-producing toxicity evaluation: the same repository, commit
  `84b12fa9a9f0af5b6bacbb663debd73d35d0d41c`.
- A-LQR paper-producing perplexity implementation: the same repository, commit
  `19fd191b79c94d66fc9f8f946ad1df8c4e59de55`.
- ITI: `likenneth/honest_llama`, commit
  `2c6b2179be7b5aa8f0a171688cf9e01b812ca327`.
- ActAdd: `montemac/activation_additions`, commit
  `cc3178cb813b640cd9644cf656d43a51e28869bd`.
- Mean/Linear-AcT: `apple/ml-act`, commit
  `d2c3560b7022da795d58f892c398ab77cff13590`.
- PID-AcT: `dungnvnus/pid-steering`, commit
  `d705c44a2f9c67e54e4d551824f46cba3c3b187e`.
- ODESteer: `ZhaoHongjue/odesteer`, commit
  `8a3c481d6493ecb3325eea5ef9c448cccfced7eb`.

The active implementations are independent package code checked by cheap
operator-level tests against these sources. Reference material is never
imported at runtime.
