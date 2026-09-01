# Residual Checks

## Caption

Residual magnitude and direction-aware propagation under the frozen Llama-3.2-1B A-LQR toxicity controller for 50 held-out RealToxicityPrompts and 50 Jigsaw prompts. Panel A shows mean layer-wise residual norm relative to the next activation norm for baseline and A-LQR passes. Panel B shows prompt-level residual amplification, with black lines marking medians; median amplification is 0.00695 for ID and 0.01578 for OOD prompts. Panels C and D compare final normalized semantic tracking error with raw residual magnitude and direction-aware residual effect, respectively. Residual magnitude is not associated with tracking error (Spearman rho = -0.069), whereas direction-aware residual effect is (rho = 0.671).

## Panel Notes

- A: Midnight-blue lines are RealToxicityPrompts (ID), dark-red lines are Jigsaw (OOD), dotted lines are baseline passes, and solid lines are A-LQR passes.
- B: Points are individual A-LQR prompts and black horizontal lines are condition medians; OOD amplification is greater in the one-sided Mann-Whitney comparison (`U = 1816`, `p = 4.84e-5`).
- C: Each point is one A-LQR prompt; the x axis is log10 stacked residual magnitude and the y axis is final semantic tracking error divided by the final-layer toxicity-feature norm.
- D: Each point is one A-LQR prompt; the x axis is log10 absolute residual-induced final semantic effect and the y axis is the same normalized tracking error.

## Checks

- Visual encodings checked against: `plots/residual_checks.pdf`, `residual_checks.py`, `README.md`, and `STYLE.md`.
- Statistics checked against: `plots/residual_checks_metrics.csv` and `plots/residual_checks_summary.json`.
- Remaining uncertainty: the result is an internal one-pass tracking diagnostic and has not yet been linked to generated-text toxicity.
