# `fig1_framing`

## Method

Conceptual comparison between closed-loop biological stimulation and closed-loop activation steering. The right panel deliberately exposes only the controller-facing abstraction: observe a hidden state, compute an intervention, and continue the model forward.

## Variables

Panel A contains neural state h_t, controller, stimulation u_t, and behavior. Panel B contains prompt/context, hidden state h_l, H∞ controller, intervention u_l, and subsequent layers/behavior.

## Statistics

None. This is a conceptual diagram.

## Legends

Teal denotes observed state, red denotes control, gray denotes context or behavior, and blue denotes the downstream model computation.

## Interpretation

The shared feedback structure motivates treating activation steering as a control problem without claiming that the biological and language-model systems are otherwise equivalent.

## Notes

Replace the box wording only if the manuscript terminology changes. No experimental values are required.

## References

Guided by the attached `Figure Arrangement.pdf` and `/home/dev/controller/paper/iclr2026_conference.tex`.

# `fig2_dynamics_and_residuals`

## Method

Four-panel summary of the existing representation-dynamics and residual analyses. The intended final figure should reuse results from the residual units rather than rerun them here.

## Variables

Panel A: two reduced state coordinates for ID, OOD, and controlled trajectories. Panel B: normalized transformer depth versus mean relative residual. Panel C: shift condition versus remaining target error as a percentage of the uncontrolled baseline. Panel D: model parameters versus OOD-minus-ID residual, stratified by early, middle, and late depth.

## Statistics

The current values are seeded synthetic placeholders. The final figure should show prompt-level distributions or uncertainty intervals and report the aggregation unit used by each source experiment.

## Legends

Blue denotes ID or controlled behavior, orange denotes natural OOD, dark red denotes adversarial OOD, and gray denotes the uncontrolled ID trajectory.

## Interpretation

The figure first establishes approximately structured representation dynamics, then shows where frozen linear control loses accuracy under distribution shift.

## Notes

Panels C and D correspond to experiments already marked done in `TODO.md`; only final result selection and harmonized plotting remain.

## References

Source units: `/home/dev/controller/code/parking/residual_checks`, `/home/dev/controller/code/parking/network_size_residual_explore`, `/home/dev/controller/code/parking/ood_explore`, and `/home/dev/controller/code/parking/ood_adversarial`.

# `fig3_robust_steerability`

## Method

Prospective prediction design. Compute S_rob = 1/gamma-star from calibration activations only, then test it against held-out OOD steering reliability across model–behavior pairs.

## Variables

Panel A: calibration-only computation pipeline. Panel B: x = S_rob and y = held-out OOD steering reliability; color identifies model family and marker identifies behavior. Panel C: cross-validated R² for scale, probe, residual, A-LQR cost, and S_rob. Panel D: observed versus predicted OOD reliability under leave-one-model-family-out validation.

## Statistics

All displayed points and bars are seeded synthetic placeholders. The final analysis should report cross-validated predictive performance, bootstrap confidence intervals, and comparison against the simpler predictors in panel C. Model–behavior pairs, not prompts from the same pair, must define the outer validation split.

## Legends

Colors identify model families; marker shapes identify behaviors. The dashed diagonal in panel D is perfect prediction.

## Interpretation

Panel B is the paper's primary “money plot.” Success means a calibration-only control quantity predicts which model–behavior pairs will remain steerable under unseen shifts.

## Notes

This is missing experiment 5 in `TODO.md`. Held-out steering outcomes must not enter A_k, B_k, D_k, gamma-star, hyperparameter selection, or feature construction.

## References

Research question and notation follow `/home/dev/controller/paper/iclr2026_conference.tex` and the frozen plan in `TODO.md`.

# `fig4_id_controller_benchmark`

## Method

In-distribution comparison of Original, A-LQR, S-PID, and H∞ across model scale and behavior benchmarks.

## Variables

Panel A: model versus truthfulness times informativeness. Panel B: model versus ID toxicity reduction. Panel C: model versus collateral utility retained. Panel D: normalized intervention energy versus ID steering reliability.

## Statistics

All bar heights and points are seeded synthetic placeholders. Final bars require the paper's fixed evaluation set and uncertainty across prompts or repeated generations. Panel D should use matched evaluation conditions for every controller.

## Legends

Gray is Original, midnight blue is A-LQR, orange is S-PID, and green is H∞. This mapping is shared by all benchmark figures.

## Interpretation

The figure asks whether robust control is competitive in distribution without hiding its intervention cost or collateral capability loss.

## Notes

This belongs to partial experiment 3. Panel D is intentionally retained because intervention effort is part of the manuscript's steering objective and the benchmark requires matched intervention energy or matched collateral degradation. Populate only full runs in the final paper; smoke runs may be shown during development but must remain visibly marked.

## References

Controller implementations and benchmark outputs live under `/home/dev/controller/code/robust_steerability` and `/home/dev/controller/code/AppliedControler`.

# `fig5_ood_controller_benchmark`

## Method

OOD comparison and crossover analysis designed to test whether robust control gains value specifically as identified dynamics become less reliable.

## Variables

Panel A: shift severity on the x-axis and steering reliability on the y-axis for Original, A-LQR, S-PID, and H∞. Panel B: model-by-OOD-condition matrix of H∞ reliability. Panel C: measured dynamics mismatch versus A-LQR and H∞ reliability. Panel D: fragility gamma-star versus the reliability advantage H∞ minus A-LQR, with marker shape identifying behavior.

## Statistics

All curves, cells, and points are seeded synthetic placeholders. The final panel D should report the association and uncertainty using held-out model–behavior pairs; repeated prompts within a pair are not independent samples for this claim.

## Legends

Gray is Original, midnight blue is A-LQR, orange is S-PID, and green is H∞. In panel D, circles, squares, and triangles denote toxicity, truthfulness, and refusal.

## Interpretation

Panels C and D are the decisive crossover test: H∞ should help most where mismatch or calibration-only fragility predicts A-LQR failure, rather than winning uniformly by arbitrary gain choice.

## Notes

Panel A/B populate partial experiment 3. Panel D is the final version of experiment 6; only a loose precursor exists. The gain sweep in missing experiment 4 must be resolved before interpreting method differences.

## References

OOD sources: `/home/dev/controller/code/parking/ood_explore` and `/home/dev/controller/code/parking/ood_adversarial`; manuscript framing: `/home/dev/controller/paper/iclr2026_conference.tex`.

# `fig6_long_context_prediction`

## Method

Long-context stress test that holds controller calibration fixed while increasing evaluation context length.

## Variables

Panel A: context length versus remaining target error for A-LQR and H∞. Panel B: context length versus collateral utility retained. Panel C: context length versus A-LQR error minus H∞ error; positive values favor H∞.

## Statistics

All curves are seeded synthetic placeholders. Final curves should include uncertainty over prompts and explicitly report the calibration context length, evaluated token budgets, and truncation policy.

## Legends

Midnight blue is A-LQR, green is H∞, and the gray horizontal line in panel C marks equal error.

## Interpretation

The figure tests the paper's high-stress prediction: robust control should become more useful when long contexts accumulate dynamics mismatch, while preserving collateral behavior.

## Notes

This is a focused extension of the long-context OOD condition. It should remain Figure 6 only if it adds a qualitatively new regime beyond Figure 5.

## References

Guided by page 7 of the attached `Figure Arrangement.pdf` and the OOD plan in `TODO.md`.

# `figs1_hinf_validation`

## Method

Numerical validation suite for the finite-horizon H∞ synthesis and its induced-gain certificate.

## Variables

Panel A: candidate attenuation gamma versus minimum feasibility margin, identifying gamma-star. Panel B: exact finite-horizon gain versus synthesized gamma-star. Panel C: disturbance energy versus performance-output energy with the certified bound. Panel D: certified tracking-error bound versus observed tracking error.

## Statistics

All values are seeded synthetic placeholders. Final points should come from controlled numerical systems with independently verified solutions and should report numerical tolerances and failure counts.

## Legends

Dashed black lines denote exact values or certificate bounds; colored points and curves denote synthesized or observed quantities.

## Interpretation

This supplement verifies that the controller and certificate are implemented correctly before language-model outcomes are used as evidence.

## Notes

Populate after the complete H∞ controller is connected. This supports implementation item 1 and should also expose memory/runtime measurements relevant to item 2.

## References

Implementation target: `/home/dev/controller/code/robust_steerability`; experiment integration: `/home/dev/controller/code/AppliedControler`.

# `figs2_nonlinearity_validation`

## Method

Stress test of local linearity as trajectories move away from the nominal calibration region and intervention strength increases.

## Variables

Panel A: distance from nominal trajectory versus linearization residual. Panel B: linear-model prediction versus observed state displacement. Panel C: transformer depth versus relative residual, colored by steering strength. Panel D: distance from nominal trajectory versus percentage of rollouts satisfying the bound, with and without a nonlinear correction.

## Statistics

All points and curves are seeded synthetic placeholders. Final analysis should state the distance metric, residual normalization, rollout unit, and uncertainty across prompts and models.

## Legends

The dashed diagonal in panel B is exact prediction. In panel C the color bar encodes intervention strength. Midnight blue is the linear certificate and green is a possible nonlinear correction.

## Interpretation

The supplement defines the regime where the local linear model and its robustness certificate remain credible, and shows how that regime contracts under stronger interventions.

## Notes

This is the lower-priority disturbance-geometry component of experiment 7. A nonlinear correction should appear only if the paper actually implements and validates one.

## References

Existing residual sources: `/home/dev/controller/code/parking/linearization_error_explore` and `/home/dev/controller/code/parking/residual_checks`.

# `figs3_tuning_and_ablations`

## Method

Supplementary tuning and ablation figure covering the two frozen experiments that should not displace the main intervention-energy comparison.

## Variables

Panel A: controller gain multiplier versus target steering success for A-LQR and H∞. Panel B: gain multiplier versus collateral utility retained and normalized intervention energy; solid lines are utility and dashed lines are energy. Panel C: calibration-set size versus relative error in gamma-star and held-out OOD prediction R². Panel D: disturbance geometry versus the reliability advantage H∞ minus A-LQR.

## Statistics

All curves and distributions are seeded synthetic placeholders. The final gain sweep must use the same evaluation prompts across gain settings and report uncertainty over prompts or repeated generations. Calibration-size results should repeat subsampling across seeds. Disturbance-geometry comparisons should use matched disturbance energy.

## Legends

Midnight blue is A-LQR and green is H∞. Solid lines in panel B denote utility retention and dashed lines denote normalized intervention energy. The gray region marks the illustrative selected gain range.

## Interpretation

Panels A–B identify an operating region that achieves steering without the utility collapse associated with oversteering. Panel C asks how much calibration data is needed for a stable prospective certificate. Panel D asks whether robust control helps specifically for disturbance directions that interact strongly with vulnerable dynamics.

## Notes

Panels A–B are missing experiment 4. Panels C–D are lower-priority experiment 7. The final selected gain must be fixed without using the held-out OOD test conditions later used for the headline comparison.

## References

The matched-energy requirement and controller ablations are specified in `/home/dev/controller/paper/iclr2026_conference.tex`; status is frozen in `TODO.md`.

# `figure_arrangement_overview`

## Method

Contact sheet assembling Figures 1–6 and Figures S1–S3 in proposed manuscript order.

## Variables

None beyond the panels documented above.

## Statistics

None. It is a layout index, and every embedded value is a placeholder inherited from the source mockup.

## Legends

Legends are inherited from each source figure.

## Interpretation

The overview makes the narrative progression explicit: framing, empirical failure mechanism, prospective predictor, ID benchmark, OOD payoff, long-context stress test, two validation supplements, and one tuning/ablation supplement.

## Notes

Run `python figs/sketch/sketch.py` from `/home/dev/controller/code` in the project environment to regenerate all PDF and PNG outputs. Replace values in the individual figure builders; do not edit the overview independently.

## References

The frozen work list is `TODO.md`; visual guidance came from the attached `Figure Arrangement.pdf`, `/home/dev/controller/paper/iclr2026_conference.tex`, and `/home/dev/controller/paper/draft sketch/main.tex`.
