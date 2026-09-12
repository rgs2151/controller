# distribution_shift_txi

The explicit data-to-figure workflow and model-specific long-context formula are documented in `PIPELINE.md`.

## Method

- Use the pinned `google/gemma-2-2b` checkpoint and the clean TruthfulQA A-LQR artifacts in `parking/bench_artifacts`.
- Reuse A-LQR's averaged 35-prompt Jacobian matrix unchanged when fitting the rank-8 full-state H∞ controller. Fit both semantic targets from the same 200 false and 200 true MC2 answer prompts; fit H∞ disturbance geometry on 200 separate generation questions.
- Select 50 evaluation questions that overlap neither the semantic-fit questions nor the disturbance questions. Apply the same question and sampling seed to A-LQR and H∞.
- Evaluate eight 50-prompt sets: ID, Spanish, Japanese-romaji, Long Context End, Long Context Start, Corrupting Words, BOS Mix, and L-CiteEval Complexity.
- Generate with the A-LQR paper settings and source generation protocol. Score every completion with the pinned TruthfulQA truth and informativeness judges, then compute `Truth (%) × Info (%) / 100`.
- Write a 1×2 grouped-bar figure separating Truth and Info, aggregate scores, and five real prompt examples from every distribution.

## Variables

- Data/input: pinned TruthfulQA generation and multiple-choice validation splits plus 50 L-CiteEval NarrativeQA/LoCoMo questions; 50 prompts per condition.
- Sessions/groups: A-LQR and corrected full-state H∞ on all eight conditions.
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
| Long Context End | Seven fixed public-domain excerpts fill about 7,168 Gemma tokens; the unchanged question is at the end. |
| Long Context Start | The same documents and question are used, but the question and `A:` cue come before the documents. |
| Corrupting Words | A frozen text-only suffix found by gradient search to maximize A-LQR overshoot on one Llama-3.2-1B prompt is appended to every TruthfulQA prompt. |
| BOS Mix | Gemma-2-2B's actual `<bos>` token is distributed through each prompt; a fixed seed assigns 16 insertions to 25 questions and 64 to the other 25. |
| L-CiteEval Complexity | Fifty NarrativeQA and LoCoMo long-context questions spanning easy, medium, and hard examples. |

## Statistics

- Tests/models: descriptive comparison of binary judge percentages; no null-hypothesis test or confidence interval is shown.
- Null hypothesis: not tested in this exploratory unit.
- Alternative hypothesis: not tested; the practical question is whether H∞ retains higher Truth and Info than A-LQR under distribution shift.
- Thresholds/decision rule: each prompt receives a binary `yes`/`no` decision from each judge, and each bar is `100 × mean(binary decision)` across the same 50 prompts.
- What the statistic means: Truth measures whether the answer is factually true; Info measures whether it is helpful or informative. Both are binary per prompt and reported as percentages across the 50 prompts.
- Why this statistic is appropriate here: separating the two paper judges reveals whether a method improves factuality, informativeness, or merely trades one against the other.

## Legends

- X axis: ID, Spanish, Japanese (romaji), Long Context End, Long Context Start, Corrupting Words, BOS Mix, and L-CiteEval Complexity.
- Left y axis: Truth percentage; higher is better.
- Right y axis: Info percentage on the same 0–100 scale; higher is better.
- Color/value: black is A-LQR and red is H∞.
- Grouping: two bars per distribution, each based on the same 50 prompts; the first seven sets share TruthfulQA anchors and L-CiteEval supplies its own 50 questions.
- Ordering/sorting: distributions use the fixed conceptual order above; methods always appear A-LQR then H∞.
- Bars/labels: bar height gives the percentage of `yes` decisions; the integer label gives the same point estimate. No error bars or confidence intervals are displayed.
- Panels: Truthfulness is on the left and Informativeness is on the right.

## Interpretation

- In this 50-prompt run, H∞ does not outperform A-LQR on Truth in any condition.
- H∞ has higher Info on ID, Spanish, Japanese-romaji, both long-context variants, and L-CiteEval Complexity, but lower Info on Corrupting Words and BOS Mix.
- The result does not support the intended claim that H∞ preserves both benchmark dimensions better than A-LQR under these shifts. The 50-prompt run is exploratory and is not a final statistical comparison.

## Notes

- `plots/ood_examples/` contains one inspection file per dataset, each with five complete literal prompts.
- `cache/datasets/` contains eight 50-row CSVs: seven matched TruthfulQA sets and one L-CiteEval set included as the eighth plotted condition.
- `lciteeval_complexity.csv` contains 25 NarrativeQA and 25 LoCoMo examples spanning easy, medium, and hard labels; the same pinned Truth and Info judges are applied to it for this analysis.
- Evaluation reads the frozen CSVs directly and never reruns translation or dataset construction; a later full-size run must use a separate frozen bundle.
- Every artifact produced by this analysis—the H∞ controller and diagnostics, shared-A copy, translations, generations, judge outputs, logs, CSVs, and plots—is stored under `parking/dist_changes/`. The unit only reads the frozen A-LQR inputs in `parking/bench_artifacts/`; it does not write to them.
- Large model artifacts, controller diagnostics, translations, and judge caches remain under ignored `cache/` storage.

## References

- `parking/bench_artifacts/`
- `parking/ood_explore/`
- `parking/ood_adversarial/`
- [L-CiteEval dataset](https://huggingface.co/datasets/Jonaszky123/L-CiteEval)
- `ref/paper_benchmark_50/prepare_data.py`
- `robust_steerability/experiments/calibration.py`
- `robust_steerability/source_methods/`
