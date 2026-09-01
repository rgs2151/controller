# Layer Residuals

## Caption

Mean layer-wise A-LQR prediction residual relative to the observed next-state norm for 50 held-out RealToxicityPrompts (RTP) and 50 held-out Jigsaw prompts. Midnight blue denotes RTP ID and dark red denotes Jigsaw OOD; dotted lines are baseline passes and solid lines are A-LQR passes.

## Panel Notes

- Single panel: normalized transformer depth runs from the earliest to final analyzed layer.

## Checks

- Visual encodings checked against: `plots/layer_residuals.pdf`, `residual_checks.py`, and `README.md`.
- Statistics checked against: descriptive layer profiles in `cache/analysis.json`.
- Remaining uncertainty: prompt means do not show prompt-level dispersion.

# Residual Amplification

## Caption

Prompt-level amplification of observed residuals through the frozen A-LQR closed loop. Points are held-out prompts and black lines are medians. Distribution shift raises median amplification from 0.00695 for RTP ID to 0.01578 for Jigsaw OOD, a 2.27-fold increase (`U=1816`, one-sided `p=4.84e-5`).

## Panel Notes

- Single panel: midnight blue denotes RTP ID and dark red denotes Jigsaw OOD.

## Checks

- Visual encodings checked against: `plots/residual_amplification.pdf`, `residual_checks.py`, and `README.md`.
- Statistics checked against: `plots/residual_checks_summary.json`.
- Remaining uncertainty: amplification is an internal prompt-pass diagnostic.

# Amplification vs Failure

## Caption

Closed-loop residual amplification versus semantic tracking failure for 50 held-out Jigsaw OOD prompts. OOD prompts with more strongly amplified residuals have larger A-LQR semantic tracking failure (Spearman `rho=0.918`, `p=7.50e-21`).

## Panel Notes

- Single panel: dark-red points are Jigsaw OOD prompts; the x axis is log10 closed-loop residual amplification.

## Checks

- Visual encodings checked against: `plots/amplification_vs_failure.pdf`, `residual_checks.py`, and `README.md`.
- Statistics checked against: `plots/residual_checks_summary.json`.
- Remaining uncertainty: this plot diagnoses A-LQR and does not contain an H-infinity controller comparison.

# Residual Magnitude vs Failure

## Caption

Stacked residual magnitude versus semantic tracking failure for 50 held-out Jigsaw OOD A-LQR prompt passes. The association is negative (Spearman `rho=-0.356`, `p=0.011`), opposite to a magnitude-only explanation of failure.

## Panel Notes

- Single panel: dark-red points are Jigsaw OOD prompts.

## Checks

- Visual encodings checked against: `plots/residual_magnitude_vs_failure.pdf`, `residual_checks.py`, and `README.md`.
- Statistics checked against: `plots/residual_checks_summary.json`.
- Remaining uncertainty: the outcome is internal tracking error rather than text toxicity.

# Direction-Aware Effect vs Failure

## Caption

Direction-aware residual effect versus semantic tracking failure for 50 held-out Jigsaw OOD A-LQR prompt passes. Direction-aware effect strongly tracks failure under distribution shift (Spearman `rho=0.922`, `p=2.23e-21`).

## Panel Notes

- Single panel: dark-red points are Jigsaw OOD prompts.

## Checks

- Visual encodings checked against: `plots/directional_effect_vs_failure.pdf`, `residual_checks.py`, and `README.md`.
- Statistics checked against: `plots/residual_checks_summary.json`.
- Remaining uncertainty: this is an internal semantic-tracking result; the generated-toxicity check is floor-limited.

# Generated-Text Toxicity

## Caption

Toxic-class probability for 100-token continuations generated from held-out prompts with matched baseline and A-LQR sampling seeds. Open circles are baseline, filled circles are A-LQR, and lines join prompt pairs. The fraction above 0.5 falls from 8% to 0% under A-LQR in both RTP ID and Jigsaw OOD.

## Panel Notes

- Single panel: midnight blue denotes RTP ID, dark red denotes Jigsaw OOD, and black bars mark medians.

## Checks

- Visual encodings checked against: `plots/generated_toxicity.pdf`, `residual_checks.py`, and `README.md`.
- Statistics checked against: `plots/generated_toxicity_summary.json`.
- Remaining uncertainty: most scores are at the classifier floor, and the mean reduction is driven by a few toxic baseline generations.

# Residual Magnitude vs Toxicity Shortfall

## Caption

Prompt-pass residual magnitude versus paired generated-text toxicity change, defined as A-LQR toxic probability minus baseline toxic probability. Residual magnitude does not predict toxicity shortfall (Spearman `rho=0.023`, `p=0.820`).

## Panel Notes

- Single panel: values below the dotted zero line indicate reduced toxicity under A-LQR; colors identify RTP ID and Jigsaw OOD.

## Checks

- Visual encodings checked against: `plots/residual_magnitude_vs_toxicity_shortfall.pdf`, `residual_checks.py`, and `README.md`.
- Statistics checked against: `plots/generated_toxicity_summary.json`.
- Remaining uncertainty: A-LQR leaves no classifier-positive generations, limiting observable behavioral failure.

# Direction-Aware Effect vs Toxicity Shortfall

## Caption

Prompt-pass direction-aware residual effect versus paired generated-text toxicity change. Direction-aware effect does not predict toxicity shortfall (Spearman `rho=-0.040`, `p=0.695`), and its correlation does not exceed the magnitude correlation (difference -0.063, bootstrap 95% CI `[-0.327, 0.218]`).

## Panel Notes

- Single panel: values below the dotted zero line indicate reduced toxicity under A-LQR; colors identify RTP ID and Jigsaw OOD.

## Checks

- Visual encodings checked against: `plots/directional_effect_vs_toxicity_shortfall.pdf`, `residual_checks.py`, and `README.md`.
- Statistics checked against: `plots/generated_toxicity_summary.json`.
- Remaining uncertainty: the result is floor-limited and does not test shifts where A-LQR retains a substantial toxic failure rate.
