# residual_checks

## Method

- Use the exact `meta-llama/Llama-3.2-1B` base checkpoint evaluated by Activation-LQR (A-LQR) for toxicity mitigation.
- Reconstruct the A-LQR toxicity feature and nominal layer dynamics from RealToxicityPrompts, then hold the controller fixed.
- Evaluate the same fixed controller on held-out RealToxicityPrompts (ID) and Jigsaw Toxic Comment prompts (OOD).
- Cache baseline and steered hidden activations for every transformer layer and every token used in each forward pass, together with token IDs, attention masks, layer controls, prompt identity, and source metadata.
- Measure each layer residual as the difference between the actual steered next-layer activation and A-LQR's frozen linear prediction.
- Propagate the observed residual sequence through the frozen A-LQR closed-loop dynamics to measure its direction-aware contribution to final semantic tracking error.
- Write a multi-panel vector PDF comparing residual magnitude, residual amplification, and their associations with final semantic tracking failure.

## Variables

- Data/input: `allenai/real-toxicity-prompts` at revision `f21629712ffd6a3d13a54fd2807ccd521c55ef74`; `tcapelle/jigsaw-toxic-comment-classification-challenge` at revision `2bf801de1b879f287943ecfc81fdca8690d9fc61`.
- Sessions/groups: A-LQR fit prompts from RealToxicityPrompts; 50 held-out ID RealToxicityPrompts prompts; 50 held-out OOD Jigsaw prompts; baseline and A-LQR rollouts for each evaluation prompt.
- Labels/targets: non-toxic minus toxic linear feature direction at every layer; feature target `beta[k] = 2.5 ||e[k]||`.
- Signals/features/measures: all-token hidden activations, last-token layer interventions, one-step linear-model residuals, stacked residual norm, residual-induced final semantic error, residual amplification, and final feature-setpoint error.
- Parameters/thresholds: model revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`; `Q = 0.1 I`, `R = I`, `Q_T = I`; deterministic seed `2151`; 200 toxic and 200 non-toxic feature-fit prompts; 50 non-toxic Jacobian-fit prompts; maximum tokenized prompt length 512.
- Outputs: `cache/prompts.json`; `cache/fit_activations.h5`; `cache/features.pt`; 50 prompt-level files under `cache/jacobians/`; `cache/controller.pt`; `cache/rollouts_{id,ood}_{baseline,alqr}.h5`; `cache/analysis.json`; and tracked `plots/residual_checks.pdf`, `plots/residual_checks.png`, `plots/residual_checks_metrics.csv`, and `plots/residual_checks_summary.json`.

## Statistics

- Tests/models: Spearman rank correlation between final semantic tracking failure and either residual magnitude or direction-aware residual effect; one-sided Mann-Whitney U comparison of whether OOD residual amplification exceeds ID; deterministic bootstrap confidence interval for the difference between the two Spearman correlations.
- Null hypothesis: direction-aware residual effect is no more associated with held-out A-LQR tracking failure than residual magnitude, and residual amplification has the same distribution for ID and OOD prompts.
- Alternative hypothesis: direction-aware residual effect is more associated with tracking failure than residual magnitude, and OOD prompts produce greater residual amplification.
- Thresholds/decision rule: report effect sizes and 95% bootstrap confidence intervals; this smoke test does not use a binary significance threshold to declare success.
- What the statistic means: Spearman correlation measures monotone prompt-level association without assuming Gaussian scores; Mann-Whitney U measures whether one condition tends to have larger amplification; the bootstrap interval quantifies uncertainty in the correlation difference.
- Why this statistic is appropriate here: the test asks whether residual direction through the fixed closed-loop dynamics adds information beyond residual size and whether that amplification changes under dataset shift.

## Legends

- X axis: panel-specific residual magnitude, direction-aware residual effect, or transformer depth.
- Y axis: final semantic tracking error, residual amplification, or mean residual norm as named on each panel.
- Color/value: midnight blue denotes held-out RealToxicityPrompts (ID); dark red denotes held-out Jigsaw prompts (OOD); black denotes pooled fitted summaries or reference annotations.
- Grouping: prompt-level points and distributions are grouped by ID versus OOD source.
- Ordering/sorting: transformer layers are ordered from embedding-side depth 0 to final-layer depth 1; categorical conditions are ordered ID then OOD.
- Lines/markers/labels: points show individual prompts; fitted rank associations and confidence summaries are labeled directly; no generated-text toxicity score is used in this internal one-pass smoke test.
- Panels: residual magnitude by layer, amplification by condition, residual magnitude versus failure, and direction-aware residual effect versus failure.

## Interpretation

- Across the 100 held-out A-LQR rollouts, residual magnitude does not track final semantic error (`Spearman rho = -0.069`, two-sided `p = 0.495`), while direction-aware residual effect does (`rho = 0.671`, `p = 2.18e-14`).
- The direction-aware-minus-magnitude correlation difference is `0.740`, with a prompt-bootstrap 95% interval of `[0.531, 0.953]`.
- Median residual amplification is `0.00695` for ID prompts and `0.01578` for OOD prompts, a 2.27-fold increase; the one-sided Mann-Whitney comparison gives `U = 1816` and `p = 4.84e-5`.
- The empirical 95th-percentile amplification is `0.02641` for ID and `0.03794` for OOD prompts.
- These results support the specific preliminary claim that the direction in which A-LQR model error enters the frozen closed loop is more informative about semantic tracking failure than residual size alone.

## Notes

- The experiment is cache-first: existing compatible assets and completed prompt groups are loaded rather than recomputed. The current local cache occupies approximately 8.4 GB.
- Hidden states are raw decoder-layer inputs plus the raw final decoder-block output, matching the A-LQR reference implementation rather than the model's post-normalization hidden state.
- Evaluation prompts are tokenized individually. All tokens used by the model are cached; inputs longer than 512 tokens are truncated to the documented cap.
- The smoke test measures one deterministic forward pass and internal semantic tracking. It does not yet claim that the same relationship predicts generated-text toxicity.

## References

- Skifstad, Yang, and Chou, “Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control,” 2026.
- `ref/lqr-activation-steering/`
- `DECISIONS.md`
- `STYLE.md`
