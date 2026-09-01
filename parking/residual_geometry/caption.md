# residual_geometry

## Caption

Representation-dynamics residual magnitude, held-out disturbance-subspace coverage, and retained calibration rank across normalized transformer depth. Panel A shows mean one-step residual norm divided by hidden-state-deviation norm for each held-out shift condition. Panel B shows the mean fraction of centered test residual energy captured by the layer-wise principal-component basis fitted on calibration prompts. Panel C shows the fraction of state dimensions required to explain at least 95% of calibration residual variance.

## Panel Notes

- A: Midnight blue, dark green, dark red, and purple lines denote in-distribution, paraphrase, out-of-distribution, and adversarial test prompts, respectively.
- B: The same colors denote mean held-out explained residual energy for each shift condition.
- C: The black line denotes retained calibration rank divided by analyzed state dimension.

## Checks

- Visual encodings checked against: `README.md`, `residual_geometry.py`, and `STYLE.md`.
- Statistics checked against: layer-wise calibration PCA and descriptive test-condition means in `README.md` and `residual_geometry.py`; no significance test.
- Remaining uncertainty: numerical results and final condition availability remain pending until the analysis-ready residual dataset is generated.
