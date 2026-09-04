# lqr_ood_failure_boxplot
## Method

- Use the frozen A-LQR controller fitted by `parking/residual_checks` for the
  pinned `meta-llama/Llama-3.2-1B` revision. No target, nominal dynamics,
  Jacobian, gain, or setpoint is refitted on the OOD prompts.
- Evaluate 50 prompts in each of nine source conditions. The ID and dataset
  conditions are unchanged held-out RealToxicityPrompts and Jigsaw prompts.
  Seven other conditions transform the same 50 held-out ID anchors, preserving
  an `anchor_id` for paired comparisons.
- Generate Spanish translations and formal domain/topic rewrites with the
  pinned `meta-llama/Llama-3.2-3B-Instruct` revision. Construct code-switching,
  pragmatic reversal, surface corruption, long-context switching, and concept
  collision deterministically from each unchanged ID source prompt.
- Run one unsteered and one frozen-LQR forward pass for every source prompt.
  Extract the last-token state at every decoder layer and the final raw decoder
  output; apply steering to the last token at all 16 decoder layers.
- Measure final target error as the absolute distance between the final state’s
  projection onto the frozen non-toxicity feature and the frozen final setpoint,
  divided by the frozen final feature norm.
- For each prompt, compute the LQR final target error divided by the unsteered
  final target error and multiply by 100. A value of 0% means complete
  correction, 100% means no LQR benefit, and values above 100% mean LQR
  increased the target error.
- Construct the tenth, adversarial OOD condition after screening by choosing,
  separately for each of the 50 ID anchors, whichever of the seven matched OOD
  transformations has the largest plotted remaining-error percentage.
- Summarize each condition with its median, interquartile range, maximum, and
  fraction above 100%, then write the prompt-level table, JSON summary, and box
  plot as PDF and PNG.

## Variables

- Data/input: 50 held-out RTP prompts and 50 held-out Jigsaw prompts from the
  existing prompt split; transformed prompt sets under `data/ood_explore/`.
- Sessions/groups: ID, dataset shift, Spanish translation, English-Spanish
  code-switching, pragmatic reversal, surface corruption, domain/topic shift,
  long-context switch, concept collision, and derived adversarial OOD.
- Labels/targets: frozen layer-wise non-toxicity feature and setpoint from the
  original RTP fit split.
- Signals/features/measures: raw last-token decoder-layer states, LQR
  interventions, normalized final target error, and remaining target error as a
  percentage of the same prompt’s unsteered error.
- Parameters/thresholds: 50 prompts per condition; seed 2151; maximum tokenized
  length 512; 16 decoder layers; no threshold for declaring a condition OOD.
- Outputs: `plots/lqr_ood_failure_boxplot.pdf`,
  `plots/lqr_ood_failure_boxplot.png`,
  `plots/lqr_ood_failure_metrics.csv`, and
  `plots/lqr_ood_failure_summary.json`.

The source conditions are defined as follows:

| Condition | Construction |
| --- | --- |
| ID | Unchanged held-out RTP prompt |
| Dataset | Unchanged held-out Jigsaw prompt |
| Spanish | Meaning- and tone-preserving Spanish rewrite generated from the matched ID prompt |
| Code-switch | Fixed bilingual English-Spanish instruction wrapped around the unchanged ID prompt |
| Pragmatic | The ID prompt is quoted inside an instruction to criticize its language |
| Corrupted | Deterministic leetspeak substitutions applied to the ID prompt |
| Domain | Formal professional, legal, technical, or academic rewrite generated from the ID prompt |
| Long ctx | The ID statement is followed by repeated neutral context and a final response instruction |
| Collision | A mathematics-and-code task is combined with the ID statement and a constructive-response instruction |
| Adversarial | Per-anchor maximum remaining error selected from the seven matched OOD transformations |

## Statistics

- Tests/models: descriptive box plots and prompt-level paired differences only;
  no inferential hypothesis test is used in this exploratory screen.
- Null hypothesis: none; the unit ranks candidate shift families rather than
  testing a preregistered population claim.
- Alternative hypothesis: none.
- Thresholds/decision rule: 100% is the mechanistic reference for no LQR
  benefit. The adversarial condition selects the largest remaining-error
  percentage among seven transformations of the same source prompt.
- What the statistic means: the median reports the typical fraction of the
  unsteered semantic target error that remains after LQR. Paired differences
  report the percentage-point increase relative to the matching ID prompt.
- Why this statistic is appropriate here: dividing by the same prompt’s
  unsteered error reduces confounding from different starting distances to the
  frozen target. Paired transformations isolate prompt shift more directly than
  unrelated dataset samples.

## Legends

- X axis: ten prompt-distribution conditions, ordered from ID through fixed OOD
  shifts to adversarial selection.
- Y axis: remaining final semantic target error after LQR, expressed as a
  percentage of the same prompt’s unsteered final target error; lower is better.
- Color/value: midnight blue is ID, dark red is each fixed OOD condition, and
  black is the derived adversarial condition.
- Grouping: 50 prompt-level observations per condition.
- Ordering/sorting: the conceptual condition order is fixed; conditions are not
  sorted by their observed medians.
- Lines/markers/labels: box center is the median, box limits are the first and
  third quartiles, whiskers extend to 1.5 interquartile ranges, and translucent
  black points are individual prompts. The annotation states that 100% would
  indicate no LQR benefit.
- Panels: one standalone box plot.

## Interpretation

- LQR leaves a median 32.47% of the unsteered final target error on held-out ID
  prompts. The ordinary Jigsaw dataset shift is only modestly worse at 34.36%.
- The strongest fixed shift is long context, with 41.06% median remaining error,
  followed by Spanish translation at 40.58% and domain/topic shift at 38.93%.
- Relative to each matched ID anchor, long context increases remaining error by
  a median 8.69 percentage points and is worse for 92% of prompts. Spanish
  translation increases it by 6.89 points and is worse for 90% of prompts.
- Adversarial selection reaches 42.89% median remaining error, a paired median
  increase of 9.60 percentage points, and is worse than ID for all 50 anchors.
  It selects long context for 22 prompts, translation for 14, domain/topic shift
  for 12, concept collision for one, and code-switching for one.
- Surface corruption does not produce the intended failure: its median is
  31.90%, slightly below ID. This shows that visible prompt distortion alone is
  not sufficient to create worse LQR tracking.
- No prompt exceeds 100%. The observed OOD effect is reduced LQR efficacy, not
  complete loss of benefit or controller-induced worsening.

## Notes

- This is an exploratory failure search, not a confirmatory OOD benchmark.
- The adversarial box is selection-biased by construction because it is chosen
  using the same outcome displayed on the y axis. It must not be reported as an
  unbiased test result. A later comparison should define the selected shift
  recipe on this screen and apply it to fresh held-out prompts.
- Spanish and domain/topic prompts are synthetic rewrites. Their semantic and
  tonal fidelity has not been independently annotated.
- The box plot measures internal semantic tracking, not generated toxicity or
  another external behavioral outcome.
- Every source rollout caches baseline and LQR last-token states across all
  layers plus the applied LQR controls, allowing later residual-direction and
  amplification analysis without rerunning the model.
- Work was split across `cuda:0` and `cuda:1`; CPU was used only for data
  assembly, summary calculations, and plotting.

## References

- `parking/residual_checks/`
- `robust_steerability/control/README.md`
- `data/README.md`
