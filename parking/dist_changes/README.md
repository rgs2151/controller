# distribution_shift_txi

The explicit data-to-figure workflow and model-specific long-context formula are documented in `PIPELINE.md`.

## Method

- Use the pinned `google/gemma-2-2b` checkpoint and the clean TruthfulQA A-LQR artifacts in `parking/bench_artifacts`.
- Reuse A-LQR's averaged 35-prompt Jacobian matrix unchanged when fitting the rank-8 full-state H∞ controller. Fit both semantic targets from the same 200 false and 200 true MC2 answer prompts; fit H∞ disturbance geometry on 200 separate generation questions.
- Select 50 evaluation questions that overlap neither the semantic-fit questions nor the disturbance questions. Apply the same question and sampling seed to A-LQR and H∞.
- Evaluate ID, Spanish, and long-context prompts plus three already-existing adversarial recipes. No new adversarial prompt is designed in this unit. The displayed adversarial condition is the existing recipe with the largest observed H∞ minus A-LQR Truth × Info difference.
- Generate with the A-LQR paper settings and source generation protocol. Score every completion with the pinned TruthfulQA truth and informativeness judges, then compute `Truth (%) × Info (%) / 100`.
- Write a 1×2 grouped-bar figure separating Truth and Info, aggregate scores, the adversarial screen, and five real prompt examples from every displayed distribution.

## Variables

- Data/input: pinned TruthfulQA generation and multiple-choice validation splits; 50 matched evaluation questions per condition.
- Sessions/groups: A-LQR and corrected full-state H∞ on ID, Spanish, long context, and one selected existing adversarial condition.
- Labels/targets: positive-minus-negative TruthfulQA MC2 semantic target; binary `True` and `Helpful` judge decisions.
- Signals/features/measures: generated completion, truth percentage, information percentage, and their product used only for adversarial-candidate selection.
- Parameters/thresholds: seed 2151 for evaluation selection; 200 false plus 200 true fit prompts; 35 shared Jacobians; 200 disturbance prompts; rank 8; 50 evaluations per method and condition; 50 generated tokens.
- Outputs: `plots/distribution_shift_truth_info.pdf`, `plots/distribution_shift_truth_info.png`, `plots/ood_examples.md`, `plots/distribution_scores.csv`, `plots/adversarial_screen.csv`, `plots/all_condition_scores.csv`, and `plots/summary.json`.

The x-axis distributions are:

| Label | Construction |
| --- | --- |
| ID | The unchanged held-out TruthfulQA prompt `Q: question A:`. |
| Spanish | A pinned Llama-3.2-3B-Instruct translation of the question, followed by the frozen request to answer in English. |
| Long context | Replacement candidate: seven coherent public-domain document excerpts, token-trimmed to 7,167–7,168 Gemma tokens, followed by the unchanged ID question. Not yet evaluated. |
| Adversarial | One of three existing recipes: D6 literal boundary-marker severity mix, A2 context saturation, or A4 role conflict. The selected recipe is named in `plots/summary.json`. |

## Statistics

- Tests/models: descriptive comparison of binary judge percentages; no null-hypothesis test or confidence interval is shown.
- Null hypothesis: not tested in this exploratory unit.
- Alternative hypothesis: not tested; the practical question is whether H∞ has higher Truth × Info than A-LQR under an existing distribution shift.
- Thresholds/decision rule: each prompt receives a binary `yes`/`no` decision from each judge. Each bar is `100 × mean(binary decision)` across the same 50 prompts. The adversarial recipe was selected using the previously reported Truth × Info product among the three frozen candidates.
- What the statistic means: Truth measures whether the answer is factually true; Info measures whether it is helpful or informative. Both are binary per prompt and reported as percentages across the 50 prompts.
- Why this statistic is appropriate here: separating the two paper judges reveals whether a method improves factuality, informativeness, or merely trades one against the other.

## Legends

- X axis: ID, Spanish, Long context, and Adversarial.
- Left y axis: Truth percentage; higher is better.
- Right y axis: Info percentage on the same 0–100 scale; higher is better.
- Color/value: black is A-LQR and red is H∞.
- Grouping: two bars per distribution, each based on the same 50 source questions.
- Ordering/sorting: distributions use the fixed conceptual order above; methods always appear A-LQR then H∞.
- Bars/labels: bar height gives the percentage of `yes` decisions; the integer label gives the same point estimate. No error bars or confidence intervals are displayed.
- Panels: Truthfulness is on the left and Informativeness is on the right.

## Interpretation

- The proposed long-context failure pattern was not observed: A-LQR scored 60.2% and H∞ scored 41.2% Truth × Info.
- H∞ also trailed A-LQR on ID (48.0% versus 56.4%) and Spanish (42.0% versus 45.6%).
- Of the three previously frozen adversarial recipes, context saturation produced the largest H∞−A-LQR difference. The gap was only +1.2 percentage points (21.8% versus 20.6%), so this run does not establish an H∞ advantage.
- The adversarial result appears to reflect severe degradation of both methods rather than robust recovery by H∞. Five real examples are documented in `plots/ood_examples.md`; complete prompt-level outputs remain in the unit-local cache.

## Notes

- This unit reuses existing adversarial constructions only. D6 is transferred exactly as the earlier literal `<|begin_of_text|>` marker recipe: 25 questions receive 16 repeats and 25 receive 64, assigned without using outcomes.
- Selecting the displayed adversarial recipe on these same 50 questions is exploratory and optimistic. A fresh disjoint confirmation set is required before treating the selected gap as population evidence.
- `plots/ood_examples.md` contains five real, matched prompts from every displayed distribution and explains their exact construction. The complete generation and judge records remain in the ignored unit-local cache rather than in large plot-level CSV files.
- `cache/datasets/` contains the four current 50-row dataset CSVs and their manifest. The existing plotted long-context value belongs to the retired repeated-sentence construction and must not be attributed to the replacement candidate.
- Every artifact produced by this analysis—the H∞ controller and diagnostics, shared-A copy, translations, generations, judge outputs, logs, CSVs, and plots—is stored under `parking/dist_changes/`. The unit only reads the frozen A-LQR inputs in `parking/bench_artifacts/`; it does not write to them.
- Large model artifacts, controller diagnostics, translations, and judge caches remain under ignored `cache/` storage.

## References

- `parking/bench_artifacts/`
- `parking/ood_explore/`
- `parking/ood_adversarial/`
- `ref/paper_benchmark_50/prepare_data.py`
- `robust_steerability/experiments/calibration.py`
- `robust_steerability/source_methods/`
