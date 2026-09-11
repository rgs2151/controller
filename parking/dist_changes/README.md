# distribution_shift_txi

## Method

- Use the pinned `google/gemma-2-2b` checkpoint and the clean TruthfulQA A-LQR artifacts in `parking/bench_artifacts`.
- Reuse A-LQR's averaged 35-prompt Jacobian matrix unchanged when fitting the rank-8 full-state H∞ controller. Fit both semantic targets from the same 200 false and 200 true MC2 answer prompts; fit H∞ disturbance geometry on 200 separate generation questions.
- Select 50 evaluation questions that overlap neither the semantic-fit questions nor the disturbance questions. Apply the same question and sampling seed to A-LQR and H∞.
- Evaluate ID, Spanish, and long-context prompts plus three already-existing adversarial recipes. No new adversarial prompt is designed in this unit. The displayed adversarial condition is the existing recipe with the largest observed H∞ minus A-LQR Truth × Info difference.
- Generate with the A-LQR paper settings and source generation protocol. Score every completion with the pinned TruthfulQA truth and informativeness judges, then compute `Truth (%) × Info (%) / 100`.
- Write the grouped bar plot, aggregate scores, adversarial screen, and prompt-level generations with raw judge decisions.

## Variables

- Data/input: pinned TruthfulQA generation and multiple-choice validation splits; 50 matched evaluation questions per condition.
- Sessions/groups: A-LQR and corrected full-state H∞ on ID, Spanish, long context, and one selected existing adversarial condition.
- Labels/targets: positive-minus-negative TruthfulQA MC2 semantic target; binary `True` and `Helpful` judge decisions.
- Signals/features/measures: generated completion, truth percentage, information percentage, their product, and bootstrap uncertainty.
- Parameters/thresholds: seed 2151 for evaluation selection; 200 false plus 200 true fit prompts; 35 shared Jacobians; 200 disturbance prompts; rank 8; 50 evaluations per method and condition; 50 generated tokens.
- Outputs: `plots/distribution_shift_txi.pdf`, `plots/distribution_shift_txi.png`, `plots/generations.csv`, `plots/distribution_scores.csv`, `plots/adversarial_screen.csv`, `plots/all_generations.csv`, `plots/all_condition_scores.csv`, and `plots/summary.json`.

The x-axis distributions are:

| Label | Construction |
| --- | --- |
| ID | The unchanged held-out TruthfulQA prompt `Q: question A:`. |
| Spanish | A pinned Llama-3.2-3B-Instruct translation of the question, followed by the frozen request to answer in English. |
| Long context | The existing neutral-archive prefix followed by an instruction to ignore it and the unchanged ID question. |
| Adversarial | One of three existing recipes: D6 literal boundary-marker severity mix, A2 context saturation, or A4 role conflict. The selected recipe is named in `plots/summary.json`. |

## Statistics

- Tests/models: descriptive comparison with a prompt bootstrap; no null-hypothesis test is used.
- Null hypothesis: not tested in this exploratory unit.
- Alternative hypothesis: not tested; the practical question is whether H∞ has higher Truth × Info than A-LQR under an existing distribution shift.
- Thresholds/decision rule: each bar is `100 × mean(True) × mean(Helpful)` over 50 prompts. Error bars are percentile 95% intervals from 10,000 prompt bootstrap resamples. The adversarial recipe maximizes the observed H∞ minus A-LQR bar among the three frozen candidates.
- What the statistic means: a high value requires the method's outputs to be both factually truthful and informative; refusals or generic empty answers lose informativeness.
- Why this statistic is appropriate here: it is the paper's behavioral TruthfulQA metric and directly tests generated answers rather than an internal steering-direction projection.

## Legends

- X axis: ID, Spanish, Long context, and Adversarial.
- Y axis: Truth × Info in percent; higher is better.
- Color/value: black is A-LQR and red is H∞.
- Grouping: two bars per distribution, each based on the same 50 source questions.
- Ordering/sorting: distributions use the fixed conceptual order above; methods always appear A-LQR then H∞.
- Lines/markers/labels: bar labels give point estimates; thin error bars give prompt-bootstrap 95% intervals.
- Panels: one standalone grouped bar chart.

## Interpretation

- The proposed long-context failure pattern was not observed: A-LQR scored 60.2% and H∞ scored 41.2% Truth × Info.
- H∞ also trailed A-LQR on ID (48.0% versus 56.4%) and Spanish (42.0% versus 45.6%).
- Of the three previously frozen adversarial recipes, context saturation produced the largest H∞−A-LQR difference. The gap was only +1.2 percentage points (21.8% versus 20.6%), with strongly overlapping bootstrap intervals, so this run does not establish an H∞ advantage.
- The adversarial result appears to reflect severe degradation of both methods rather than robust recovery by H∞. The prompt-level outputs in `plots/generations.csv` should be inspected before deciding whether this distribution is useful for a confirmatory experiment.

## Notes

- This unit reuses existing adversarial constructions only. D6 is transferred exactly as the earlier literal `<|begin_of_text|>` marker recipe: 25 questions receive 16 repeats and 25 receive 64, assigned without using outcomes.
- Selecting the displayed adversarial recipe on these same 50 questions is exploratory and optimistic. A fresh disjoint confirmation set is required before treating the selected gap as population evidence.
- `plots/generations.csv` contains the exact 400 displayed prompt/method rows. `plots/all_generations.csv` also retains the nonselected adversarial candidates.
- Every artifact produced by this analysis—the H∞ controller and diagnostics, shared-A copy, translations, generations, judge outputs, logs, CSVs, and plots—is stored under `parking/dist_changes/`. The unit only reads the frozen A-LQR inputs in `parking/bench_artifacts/`; it does not write to them.
- Large model artifacts, controller diagnostics, translations, and judge caches remain under ignored `cache/` storage.

## References

- `parking/bench_artifacts/`
- `parking/ood_explore/`
- `parking/ood_adversarial/`
- `ref/paper_benchmark_50/prepare_data.py`
- `robust_steerability/experiments/calibration.py`
- `robust_steerability/source_methods/`
