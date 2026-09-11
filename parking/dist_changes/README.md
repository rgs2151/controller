# distribution_shift_txi

The explicit data-to-figure workflow and model-specific long-context formula are documented in `PIPELINE.md`.

## Method

- Use the pinned `google/gemma-2-2b` checkpoint and the clean TruthfulQA A-LQR artifacts in `parking/bench_artifacts`.
- Reuse A-LQR's averaged 35-prompt Jacobian matrix unchanged when fitting the rank-8 full-state H∞ controller. Fit both semantic targets from the same 200 false and 200 true MC2 answer prompts; fit H∞ disturbance geometry on 200 separate generation questions.
- Select 50 evaluation questions that overlap neither the semantic-fit questions nor the disturbance questions. Apply the same question and sampling seed to A-LQR and H∞.
- Evaluate the matched ID, Spanish, Japanese-romaji, long-context, D2, D3, and D6 sets; D2/D3/D6 reuse completed attacks rather than designing new ones.
- Generate with the A-LQR paper settings and source generation protocol. Score every completion with the pinned TruthfulQA truth and informativeness judges, then compute `Truth (%) × Info (%) / 100`.
- Write a 1×2 grouped-bar figure separating Truth and Info, aggregate scores, and five real prompt examples from every distribution.

## Variables

- Data/input: pinned TruthfulQA generation and multiple-choice validation splits; 50 matched evaluation questions per condition.
- Sessions/groups: A-LQR and corrected full-state H∞ on ID, Spanish, Japanese romaji, long context, D2, D3, and D6.
- Labels/targets: positive-minus-negative TruthfulQA MC2 semantic target; binary `True` and `Helpful` judge decisions.
- Signals/features/measures: generated completion, truth percentage, information percentage, and their product as a secondary aggregate.
- Parameters/thresholds: seed 2151; 200 false plus 200 true fit prompts; 35 shared Jacobians; 200 disturbance prompts; rank 8; 50 evaluations per method and condition; 50 generated tokens.
- Outputs: `plots/distribution_shift_truth_info.pdf`, `plots/distribution_shift_truth_info.png`, one file per dataset under `plots/ood_examples/`, `plots/distribution_scores.csv`, and `plots/summary.json`.

The x-axis distributions are:

| Label | Construction |
| --- | --- |
| ID | The unchanged held-out TruthfulQA prompt `Q: question A:`. |
| Spanish | Llama-3.2-3B-Instruct deterministically translates the question into Spanish, followed by a request to answer in English. |
| Japanese (romaji) | Meta-Llama-3.1-8B-Instruct deterministically translates the question into Japanese, pykakasi converts it to Hepburn romaji, and the prompt asks for an English answer. |
| Long context | Seven deterministic public-domain book excerpts are token-trimmed to about 7,168 Gemma tokens and placed before the unchanged question. |
| D2 | Gradient search found one text-only suffix that maximized A-LQR overshoot on a single Llama-3.2-1B prompt; that exact frozen suffix is appended to every TruthfulQA prompt. |
| D3 | Starting from D2, one text-only suffix was jointly refined against the four Llama-3.2-1B prompts where D2 transferred weakest; that exact frozen suffix is appended to every TruthfulQA prompt. |
| D6 | Gemma-2-2B's actual `<bos>` token is distributed evenly through each Q/A prompt, with seed 2151 assigning 16 insertions to 25 questions and 64 to the other 25 without using outcomes. |

## Statistics

- Tests/models: descriptive comparison of binary judge percentages; no null-hypothesis test or confidence interval is shown.
- Null hypothesis: not tested in this exploratory unit.
- Alternative hypothesis: not tested; the practical question is whether H∞ retains higher Truth and Info than A-LQR under distribution shift.
- Thresholds/decision rule: each prompt receives a binary `yes`/`no` decision from each judge, and each bar is `100 × mean(binary decision)` across the same 50 prompts.
- What the statistic means: Truth measures whether the answer is factually true; Info measures whether it is helpful or informative. Both are binary per prompt and reported as percentages across the 50 prompts.
- Why this statistic is appropriate here: separating the two paper judges reveals whether a method improves factuality, informativeness, or merely trades one against the other.

## Legends

- X axis: ID, Spanish, Japanese (romaji), Long context, D2, D3, and D6.
- Left y axis: Truth percentage; higher is better.
- Right y axis: Info percentage on the same 0–100 scale; higher is better.
- Color/value: black is A-LQR and red is H∞.
- Grouping: two bars per distribution, each based on the same 50 source questions.
- Ordering/sorting: distributions use the fixed conceptual order above; methods always appear A-LQR then H∞.
- Bars/labels: bar height gives the percentage of `yes` decisions; the integer label gives the same point estimate. No error bars or confidence intervals are displayed.
- Panels: Truthfulness is on the left and Informativeness is on the right.

## Interpretation

- The current plotted scores predate this finalized seven-set dataset layout and are retained only as the previous iteration.
- No comparison among the new Japanese-romaji, D2, D3, D6, or replacement long-context sets is claimed until reevaluation.

## Notes

- D2, D3, and D6 are temporary names retained so their prompt sets can be inspected before final naming.
- `plots/ood_examples/` contains one inspection file per dataset, each with five complete matched prompts.
- `cache/datasets/` contains the seven current 50-row dataset CSVs and their manifest; the existing plot must not be attributed to this unevaluated layout.
- Evaluation reads the frozen CSVs directly and never reruns translation or dataset construction; a later full-size run must use a separate frozen bundle.
- Every artifact produced by this analysis—the H∞ controller and diagnostics, shared-A copy, translations, generations, judge outputs, logs, CSVs, and plots—is stored under `parking/dist_changes/`. The unit only reads the frozen A-LQR inputs in `parking/bench_artifacts/`; it does not write to them.
- Large model artifacts, controller diagnostics, translations, and judge caches remain under ignored `cache/` storage.

## References

- `parking/bench_artifacts/`
- `parking/ood_explore/`
- `parking/ood_adversarial/`
- `ref/paper_benchmark_50/prepare_data.py`
- `robust_steerability/experiments/calibration.py`
- `robust_steerability/source_methods/`
