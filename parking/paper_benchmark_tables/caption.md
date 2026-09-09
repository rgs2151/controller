# table1

## Caption

Toxicity steering with 50 prompts per condition and frozen settings. Toxicity is the percentage of continuations assigned the toxic classifier label and is shown with Bernoulli standard error. Dist-2 is pooled ID bigram diversity, including adjacent completion boundaries. PPL scores the jointly tokenized prompt and continuation, right-truncated to 128 tokens, under a fixed unsteered Mistral-7B. MMLU uses 50 context-fitting, intact five-shot questions and one greedy answer token. Spanish requests English output; Adversarial transfers D6 literal markers; Jigsaw/Long use longest Jigsaw/ToxicChat prompts. Baseline strengths were frozen by calibration-only preflight. Red TBD cells are unmeasured.

## Panel Notes

- Single table: five model blocks, ten methods, eight outcome columns.

## Checks

- Visual encodings checked against: table exporter and existing table layout.
- Statistics checked against: current runner, quality scorer, and unit READMEs.
- Remaining uncertainty: benchmark jobs and quality scoring must finish before final cell coverage is known.

# table2

## Caption

Truthfulness steering with separate truthfulness calibration, fixed settings, and 50 questions per condition. T·I is the product of marginal True and Info rates (%); its SE includes their within-question covariance. True, Info, and MMLU show percentage mean ± Bernoulli SE. Spanish requests English output; Adversarial transfers D6 literal markers; Long adds unrelated archive text. MMLU uses the same intact five-shot questions. Baseline operators use our shared fit/intervention setup. Red TBD cells are unmeasured. These are single-run prompt-level errors, not variation across repeated runs.

## Panel Notes

- Single table: five model blocks, ten methods, seven outcome columns.

## Checks

- Visual encodings checked against: table exporter and existing table layout.
- Statistics checked against: current runner and unit READMEs.
- Remaining uncertainty: truthfulness jobs have not all completed; caption makes no outcome claim.
