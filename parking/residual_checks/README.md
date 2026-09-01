# layer_residuals

## Method

- Use `meta-llama/Llama-3.2-1B` with the A-LQR toxicity feature, nominal trajectory, Jacobians, gains, and setpoint fit from disjoint RealToxicityPrompts records.
- Run the frozen controller and an unsteered baseline on 50 held-out RealToxicityPrompts prompts and 50 held-out Jigsaw prompts.
- At every transformer layer, subtract A-LQR's one-step prediction from the observed next-layer raw decoder state and divide the residual norm by the observed next-state norm.
- Average the relative residual separately by dataset and controller and write `plots/layer_residuals.pdf` and `plots/layer_residuals.png`.

## Variables

- Data/input: held-out RealToxicityPrompts (RTP) and Jigsaw prompts from `cache/prompts.json`.
- Sessions/groups: RTP ID versus Jigsaw OOD; baseline versus A-LQR.
- Labels/targets: normalized layer depth and one-step A-LQR residual.
- Signals/features/measures: raw decoder-layer inputs and raw final decoder-block output at the last prompt token.
- Parameters/thresholds: 50 prompts per dataset; seed 2151; Llama-3.2-1B revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
- Outputs: `plots/layer_residuals.pdf` and `plots/layer_residuals.png`.

## Statistics

- Tests/models: no inferential test; lines are prompt means at each layer.
- Null hypothesis: none; this output is descriptive.
- Alternative hypothesis: none; this output is descriptive.
- Thresholds/decision rule: none.
- What the statistic means: each point is the mean residual norm relative to the observed next-state norm.
- Why this statistic is appropriate here: relative normalization makes layer-wise model mismatch readable without raw activation scale dominating.

## Legends

- X axis: transformer depth normalized from 0 to 1.
- Y axis: mean `||xi[k]|| / ||x[k+1]||`.
- Color/value: midnight blue is RTP ID; dark red is Jigsaw OOD.
- Grouping: dataset and controller.
- Ordering/sorting: transformer layer order.
- Lines/markers/labels: dotted is baseline; solid is A-LQR.
- Panels: one standalone plot.

## Interpretation

- Relative one-step mismatch is present throughout the network and is generally larger for Jigsaw OOD prompts.

## Notes

- The states use the upstream-compatible raw decoder coordinates, not the post-final-normalization hidden state.

## References

- Skifstad, Yang, and Chou, “Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control,” 2026.
- `ref/lqr-activation-steering/`

# residual_amplification

## Method

- Compute each held-out A-LQR residual sequence `xi[k]` from the frozen nominal trajectory and mean Jacobians.
- Propagate that sequence through the frozen A-LQR closed-loop dynamics and project the final response onto the non-toxicity feature.
- Divide the absolute projected effect by the stacked residual norm to obtain prompt-level residual amplification.
- Write `plots/residual_amplification.pdf` and `plots/residual_amplification.png`.

## Variables

- Data/input: 50 RTP ID and 50 Jigsaw OOD A-LQR prompt rollouts.
- Sessions/groups: ID versus OOD prompts.
- Labels/targets: final non-toxicity feature direction.
- Signals/features/measures: stacked residual norm, closed-loop residual response, and residual amplification `|T_LQR xi| / ||xi||`.
- Parameters/thresholds: frozen `Q=0.1I`, `R=I`, `Q_T=I`, and setpoint multiplier 2.5.
- Outputs: `plots/residual_amplification.pdf` and `plots/residual_amplification.png`.

## Statistics

- Tests/models: one-sided Mann-Whitney U test of OOD amplification greater than ID amplification.
- Null hypothesis: OOD prompts do not tend to have greater amplification than ID prompts.
- Alternative hypothesis: OOD prompts tend to have greater amplification.
- Thresholds/decision rule: report the effect and p-value without a binary significance gate.
- What the statistic means: U ranks whether OOD prompt-level amplification tends to exceed ID amplification.
- Why this statistic is appropriate here: amplification is skewed and the two prompt groups are independent.

## Legends

- X axis: RTP ID and Jigsaw OOD.
- Y axis: residual amplification `|T_LQR xi| / ||xi||`.
- Color/value: midnight blue is RTP ID; dark red is Jigsaw OOD.
- Grouping: dataset.
- Ordering/sorting: ID then OOD.
- Lines/markers/labels: points are prompts; black horizontal lines are medians; the annotation reports the OOD/ID median ratio and one-sided Mann-Whitney p-value.
- Panels: one standalone plot.

## Interpretation

- Distribution shift raises median closed-loop residual amplification from 0.00695 ID to 0.01578 OOD, a 2.27-fold increase (`U=1816`, one-sided `p=4.84e-5`).

## Notes

- The empirical 95th-percentile amplification is 0.02641 ID and 0.03794 OOD.

## References

- `plots/residual_checks_summary.json`
- `ref/lqr-activation-steering/`

# amplification_vs_failure

## Method

- Restrict the held-out A-LQR evaluation to the 50 Jigsaw OOD prompts.
- Compute each prompt's closed-loop residual amplification as the direction-aware final residual effect divided by the stacked residual norm.
- Compare amplification with final normalized semantic tracking failure and write a standalone scatter plot.

## Variables

- Data/input: 50 held-out Jigsaw OOD A-LQR prompt rollouts.
- Sessions/groups: one OOD condition.
- Labels/targets: final normalized non-toxicity tracking failure.
- Signals/features/measures: `log10` closed-loop residual amplification and semantic tracking failure.
- Parameters/thresholds: epsilon `1e-12` before the logarithm; no missing records.
- Outputs: `plots/amplification_vs_failure.pdf` and `plots/amplification_vs_failure.png`.

## Statistics

- Tests/models: OOD-only Spearman rank correlation.
- Null hypothesis: residual amplification has no monotone association with semantic tracking failure under OOD shift.
- Alternative hypothesis: greater residual amplification accompanies greater semantic tracking failure under OOD shift.
- Thresholds/decision rule: report rho and the two-sided p-value; a large positive rho supports the proposed failure mechanism.
- What the statistic means: rho measures whether OOD prompts with larger amplification tend to have larger semantic tracking failure.
- Why this statistic is appropriate here: the claim concerns prompt ordering under a shifted distribution and does not require a linear relationship.

## Legends

- X axis: `log10` closed-loop residual amplification.
- Y axis: final normalized semantic tracking failure.
- Color/value: dark red denotes Jigsaw OOD prompts.
- Grouping: one OOD prompt set.
- Ordering/sorting: none.
- Lines/markers/labels: each point is one prompt; the title and annotation report rho and p.
- Panels: one standalone plot.

## Interpretation

- Under OOD shift, A-LQR residual amplification strongly tracks semantic tracking failure (`rho=0.918`, `p=7.50e-21`).
- Together with the 2.27-fold OOD amplification increase, this directly supports the smoke-test claim that shifted residual directions are amplified into semantic failure.

## Notes

- This establishes the failure mechanism targeted by H-infinity control; it does not yet compare an implemented H-infinity controller against A-LQR.

## References

- `plots/residual_checks_metrics.csv`
- `plots/residual_checks_summary.json`
- `ref/lqr-activation-steering/`

# residual_magnitude_vs_failure

## Method

- Restrict the evaluation to the 50 held-out Jigsaw OOD A-LQR prompts, stack each prompt's layer residuals, and compute their Euclidean norm.
- Measure final internal tracking failure as the absolute final non-toxicity setpoint error divided by the final feature norm.
- Compare prompt-level residual magnitude with internal tracking failure and write a standalone scatter plot.

## Variables

- Data/input: 50 held-out Jigsaw OOD A-LQR prompt rollouts.
- Sessions/groups: one OOD condition.
- Labels/targets: normalized final non-toxicity tracking error.
- Signals/features/measures: `log10 ||xi||` and normalized internal tracking failure.
- Parameters/thresholds: epsilon `1e-12` before the logarithm.
- Outputs: `plots/residual_magnitude_vs_failure.pdf` and `plots/residual_magnitude_vs_failure.png`.

## Statistics

- Tests/models: OOD-only Spearman rank correlation.
- Null hypothesis: residual magnitude has no monotone association with internal tracking failure.
- Alternative hypothesis: residual magnitude has a monotone association with failure.
- Thresholds/decision rule: report rho and the two-sided p-value; no binary success threshold.
- What the statistic means: rho measures whether prompts with larger residual norms tend to have larger failure.
- Why this statistic is appropriate here: it does not assume Gaussian variables or a linear relationship.

## Legends

- X axis: `log10 ||xi||`.
- Y axis: final normalized internal tracking error.
- Color/value: dark red denotes Jigsaw OOD prompts.
- Grouping: one OOD prompt set.
- Ordering/sorting: none.
- Lines/markers/labels: each point is one prompt.
- Panels: one standalone plot.

## Interpretation

- Residual magnitude is negatively associated with OOD semantic failure (`rho=-0.356`, two-sided `p=0.011`), opposite to the magnitude-only explanation that larger residuals should cause larger failure.

## Notes

- This is the magnitude-only negative control for the OOD amplification result.

## References

- `plots/residual_checks_metrics.csv`
- `plots/residual_checks_summary.json`

# directional_effect_vs_failure

## Method

- Propagate each observed residual sequence through the frozen A-LQR closed-loop dynamics.
- Project the final propagated response onto the non-toxicity feature and normalize by the final feature norm.
- Compare this direction-aware effect with final internal tracking failure and write a standalone scatter plot.

## Variables

- Data/input: the same 50 held-out Jigsaw OOD A-LQR prompt rollouts used for the magnitude-only control.
- Sessions/groups: one OOD condition.
- Labels/targets: normalized final non-toxicity tracking error.
- Signals/features/measures: `log10 |T_LQR xi|` and normalized internal tracking failure.
- Parameters/thresholds: epsilon `1e-12` before the logarithm; no missing records.
- Outputs: `plots/directional_effect_vs_failure.pdf` and `plots/directional_effect_vs_failure.png`.

## Statistics

- Tests/models: OOD-only Spearman rank correlation.
- Null hypothesis: direction-aware residual effect has no monotone association with semantic tracking failure under OOD shift.
- Alternative hypothesis: greater direction-aware residual effect accompanies greater semantic tracking failure under OOD shift.
- Thresholds/decision rule: report rho and the two-sided p-value; a large positive rho supports the proposed direction-dependent failure mechanism.
- What the statistic means: rho measures whether OOD prompts with larger residual effect in the semantic direction tend to have larger failure.
- Why this statistic is appropriate here: it tests the direction-dependent mechanism directly within the shifted prompt distribution.

## Legends

- X axis: `log10 |T_LQR xi|`.
- Y axis: final normalized internal tracking error.
- Color/value: dark red denotes Jigsaw OOD prompts.
- Grouping: one OOD prompt set.
- Ordering/sorting: none.
- Lines/markers/labels: each point is one prompt.
- Panels: one standalone plot.

## Interpretation

- Under OOD shift, direction-aware residual effect strongly tracks semantic failure (`rho=0.922`, `p=2.23e-21`).

## Notes

- The generated-toxicity result remains a separate floor-limited downstream check; it does not alter this internal smoke-test result.

## References

- `plots/residual_checks_metrics.csv`
- `plots/residual_checks_summary.json`

# generated_toxicity

## Method

- Generate one 100-token continuation per held-out prompt under baseline and frozen A-LQR with matched per-prompt random seeds.
- Match the upstream toxicity evaluation settings: sampling enabled, `top_p=0.3`, temperature 1.0, and repetition penalty 1.2.
- Score only the generated continuation with `s-nlp/roberta_toxicity_classifier` revision `048c25bb1e199b98802784f96325f4840f22145d` and extract the toxic-class probability.
- Connect each prompt's paired baseline and A-LQR scores and write `plots/generated_toxicity.pdf` and `plots/generated_toxicity.png`.

## Variables

- Data/input: 50 held-out RTP ID prompts and 50 held-out Jigsaw OOD prompts.
- Sessions/groups: dataset and controller, paired by prompt and sampling seed.
- Labels/targets: classifier toxic probability for the generated continuation only.
- Signals/features/measures: baseline and A-LQR toxicity probability, toxic fraction at probability 0.5, and paired toxicity reduction.
- Parameters/thresholds: 100 maximum new tokens; seed `2151 + prompt_index`, with an OOD offset of 100,000; classifier truncation at 512 tokens.
- Outputs: `plots/generated_toxicity.pdf`, `plots/generated_toxicity.png`, `plots/generated_toxicity_metrics.csv`, and `plots/generated_toxicity_summary.json`.

## Statistics

- Tests/models: one-sided paired Wilcoxon signed-rank test within each dataset.
- Null hypothesis: baseline toxic probability is not greater than A-LQR toxic probability.
- Alternative hypothesis: baseline toxic probability is greater than A-LQR toxic probability.
- Thresholds/decision rule: report paired effect summaries and p-values without using them as a paper-level success gate.
- What the statistic means: the signed-rank test asks whether paired prompt-level toxicity scores tend to decrease under A-LQR.
- Why this statistic is appropriate here: baseline and A-LQR generations are paired by prompt and seed, and probabilities are highly skewed.

## Legends

- X axis: baseline and A-LQR within RTP ID and Jigsaw OOD.
- Y axis: toxic-class probability from 0 to 1.
- Color/value: midnight blue is RTP ID; dark red is Jigsaw OOD.
- Grouping: paired prompts within dataset.
- Ordering/sorting: RTP baseline, RTP A-LQR, Jigsaw baseline, Jigsaw A-LQR.
- Lines/markers/labels: light lines connect paired generations; open circles are baseline; filled circles are A-LQR; black bars are medians.
- Panels: one standalone plot.

## Interpretation

- ID mean toxic probability falls from 0.0800 to 0.000039 (`p=3.97e-6`), and OOD falls from 0.0891 to 0.00293 (`p=1.22e-4`).
- The fraction classified toxic at 0.5 falls from 8% to 0% in both datasets.

## Notes

- Most scores are already close to zero; the mean reduction is driven by a few high-toxicity baseline generations.
- `cache/generations_{id,ood}_{baseline,alqr}.h5` stores prompt and generated token IDs, completion text, every token's raw state at all layers, and every applied control. A causal teacher-forced replay stores each token once instead of duplicating prefixes at every decoding step.
- The complete reusable unit cache, including fit activations, Jacobians, prompt rollouts, generations, and toxicity scores, occupies approximately 11 GB locally and remains ignored by git.

## References

- `ref/lqr-activation-steering/steer/toxicity/test_toxicity.py`
- `plots/generated_toxicity_summary.json`

# residual_magnitude_vs_toxicity_shortfall

## Method

- Define behavioral toxicity shortfall as A-LQR toxic probability minus the matched baseline toxic probability for each prompt.
- Join that paired outcome to the prompt's pre-generation residual magnitude by `prompt_id` and dataset.
- Compare residual magnitude with behavioral shortfall and write a standalone scatter plot.

## Variables

- Data/input: 100 paired baseline/A-LQR generations joined to the 100 prompt rollouts.
- Sessions/groups: RTP ID and Jigsaw OOD.
- Labels/targets: toxicity change, where positive values mean A-LQR increased toxicity and negative values mean it reduced toxicity.
- Signals/features/measures: `log10 ||xi||` and A-LQR-minus-baseline toxic probability.
- Parameters/thresholds: exact prompt-ID join; no missing records or imputation.
- Outputs: `plots/residual_magnitude_vs_toxicity_shortfall.pdf` and `plots/residual_magnitude_vs_toxicity_shortfall.png`.

## Statistics

- Tests/models: pooled Spearman rank correlation.
- Null hypothesis: residual magnitude has no monotone association with generated-text toxicity shortfall.
- Alternative hypothesis: residual magnitude has a monotone association with shortfall.
- Thresholds/decision rule: report rho and two-sided p-value; no binary success gate.
- What the statistic means: positive rho would mean larger residuals tend to accompany worse paired toxicity outcomes.
- Why this statistic is appropriate here: both variables are skewed and the question concerns prompt ordering rather than a linear slope.

## Legends

- X axis: `log10 ||xi||` from the A-LQR prompt pass.
- Y axis: A-LQR toxic probability minus baseline toxic probability.
- Color/value: midnight blue is RTP ID; dark red is Jigsaw OOD.
- Grouping: dataset.
- Ordering/sorting: none.
- Lines/markers/labels: each point is one paired prompt; the dotted horizontal line marks no toxicity change.
- Panels: one standalone plot.

## Interpretation

- Residual magnitude does not predict generated-text toxicity shortfall (`rho=0.023`, `p=0.820`).

## Notes

- The outcome has a strong floor because A-LQR leaves no samples above the classifier's 0.5 toxic threshold.

## References

- `plots/generated_toxicity_metrics.csv`
- `plots/generated_toxicity_summary.json`

# directional_effect_vs_toxicity_shortfall

## Method

- Join the prompt-level direction-aware residual effect to the matched A-LQR-minus-baseline generated toxicity change.
- Compare prompt ordering under the direction-aware metric with ordering under behavioral toxicity shortfall.
- Compare this correlation against the residual-magnitude correlation with a deterministic prompt bootstrap.
- Write `plots/directional_effect_vs_toxicity_shortfall.pdf` and `plots/directional_effect_vs_toxicity_shortfall.png`.

## Variables

- Data/input: the same 100 paired generations and prompt rollouts.
- Sessions/groups: RTP ID and Jigsaw OOD.
- Labels/targets: A-LQR-minus-baseline toxic probability.
- Signals/features/measures: `log10 |T_LQR xi|`, residual amplification, and toxicity shortfall.
- Parameters/thresholds: 2,000 bootstrap samples; no imputation.
- Outputs: `plots/directional_effect_vs_toxicity_shortfall.pdf` and `plots/directional_effect_vs_toxicity_shortfall.png`.

## Statistics

- Tests/models: pooled Spearman correlations, a bootstrap 95% interval for directional-effect rho minus magnitude rho, and a one-sided Mann-Whitney test of OOD shortfall greater than ID.
- Null hypothesis: direction-aware effect is no more associated with toxicity shortfall than residual magnitude, and OOD shortfall does not tend to exceed ID.
- Alternative hypothesis: direction-aware effect is more associated with shortfall, and OOD shortfall tends to exceed ID.
- Thresholds/decision rule: a correlation-difference interval excluding zero would support better behavioral prediction.
- What the statistic means: rho measures prompt ordering; the bootstrap compares predictors; Mann-Whitney compares dataset-shift outcomes.
- Why this statistic is appropriate here: it tests the original direction-aware claim against the magnitude-only baseline using the same behavioral outcome.

## Legends

- X axis: `log10 |T_LQR xi|` from the A-LQR prompt pass.
- Y axis: A-LQR toxic probability minus baseline toxic probability.
- Color/value: midnight blue is RTP ID; dark red is Jigsaw OOD.
- Grouping: dataset.
- Ordering/sorting: none.
- Lines/markers/labels: each point is one paired prompt; the dotted horizontal line marks no toxicity change.
- Panels: one standalone plot.

## Interpretation

- Direction-aware effect does not predict generated-text toxicity shortfall (`rho=-0.040`, `p=0.695`); residual amplification is also null (`rho=-0.031`, `p=0.759`).
- Direction-aware rho minus magnitude rho is -0.063 with bootstrap 95% CI `[-0.327, 0.218]`, and OOD shortfall is not greater than ID (`p=0.225`).
- The internal direction-aware result therefore does not transfer to generated toxicity in this small, floor-limited smoke test.

## Notes

- This is a negative behavioral-prediction result, not evidence that residual direction is irrelevant under harder shifts or controllers with nonzero failure rates.

## References

- `plots/generated_toxicity_metrics.csv`
- `plots/generated_toxicity_summary.json`
- `DECISIONS.md`
