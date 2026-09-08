# table1

## Caption

Toxicity steering with 50 prompts per condition and fixed settings. Toxicity is mean toxic-class probability (%), not thresholded frequency. Values show mean ± prompt-level SE; Dist-2 is pooled ID bigram diversity, without cross-completion bigrams. PPL scores nonempty ID continuations under a fixed unsteered Mistral-7B; a smaller valid count is shown explicitly. MMLU uses 50 context-fitting, intact five-shot questions. Spanish requests English output; Adversarial transfers D6 literal markers; Jigsaw/Long use longest Jigsaw/ToxicChat prompts. Baseline operators use our shared fit/intervention setup, not the original papers' complete protocols. Red TBD cells are unmeasured or undefined.

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
