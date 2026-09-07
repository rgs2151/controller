# Control-Theory Collaborator Workplan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-07
- Verification Status: UNVERIFIED
- Version Label: code_plan_v2

## Rule

There are no jointly owned experiments or figures. Each owner runs every model, baseline, metric, and analysis needed for their own figures.

The visual specification in `figs/sketch/` is frozen at commit `415c45b`. It is read-only. Real results must be written to new result units.

## Collaborator owns the complete robust-control track

She owns implementation, LLM execution, analysis, and final plotting for Figures 3, 5, 6, S1, S2, and S3. She must also run Original, A-LQR, and S-PID herself wherever those baselines are required in her figures.

### A. Complete and verify H∞

Primary files:

- `robust_steerability/control/h_infinity.py`
- `robust_steerability/control/metrics.py`
- `robust_steerability/control/types.py`
- `tests/test_control.py`

Work:

1. Audit the finite-horizon minimax recursion, sign convention, terminal cost, and feasibility conditions.
2. Validate scalar, MIMO, rectangular-control, rectangular-disturbance, and time-varying systems.
3. Compare gamma-star against the independent induced-gain calculation on small systems.
4. Add near-singular, infeasible, serialization, dtype, and CPU/GPU tests.
5. Optimize synthesis memory and runtime without changing the numerical result.
6. Report runtime and peak memory across horizon, state dimension, and disturbance rank.

### B. Connect H∞ to the experiment pipeline

Primary entry points:

- `AppliedControler/run_steering.py`
- `AppliedControler/method_registry.py`
- `robust_steerability/runtime/policy.py`
- `robust_steerability/calibration/disturbances.py`

She owns this integration completely. She should use the existing model-loading, activation-hook, calibration, generation, and evaluation code rather than build a new Hugging Face pipeline from scratch.

Required correction: `AppliedControler/run_steering.py` currently passes the identity control channel as the H∞ disturbance channel. Replace that placeholder with disturbance geometry estimated from calibration residuals.

The finished pipeline must:

1. Load the selected Hugging Face model and tokenizer.
2. Collect calibration activations and Jacobians using existing helpers.
3. Fit A, B, and calibration-residual disturbance geometry D.
4. Construct the finite-horizon control problem.
5. Synthesize H∞ and save gains, feasibility, gamma-star, S_rob, and diagnostics.
6. Apply the returned interventions during generation.
7. Cache calibration and controller artifacts so sweeps do not repeat model extraction.
8. Record controller parameters, model revision, prompt split, seed, dtype, device, and configuration hash.

### C. Own Figure S1 — H∞ validation

This is her easiest first complete figure and requires no language model:

- A: feasibility margin versus candidate gamma.
- B: synthesized gamma-star versus an independent exact result.
- C: performance-output energy versus disturbance energy and its bound.
- D: observed tracking error versus its certified bound.

Deliver a new real-result unit under `figs/`; do not edit `figs/sketch/`.

### D. Own Figure S3 — tuning and ablations

She runs and plots all four panels:

- A: H∞ and A-LQR tuning sweep versus target steering success.
- B: oversteering, utility retention, and normalized intervention energy.
- C: calibration-size ablation for gamma-star stability and OOD prediction.
- D: energy-matched disturbance geometries: isotropic, learned residual, target-aligned, and worst-case.

For H∞, do not multiply a synthesized gain afterward if that breaks the certificate. Resynthesize across a mathematically valid tuning parameter such as fixed Q/R/S choices.

### E. Own Figure 3 — the robust-steerability money experiment

She performs the experiment end-to-end:

1. Select the model–behavior pairs specified by the paper configuration.
2. Build gamma-star and S_rob using calibration prompts only.
3. Run held-out OOD steering for every pair.
4. Compute competing predictors: model scale, probe quality, residual error, and nominal LQR cost.
5. Fit the cross-validated prediction analysis.
6. Run leave-one-model-family-out evaluation.
7. Produce every panel of Figure 3.

Held-out OOD outcomes must never enter controller construction, parameter tuning, or S_rob computation.

### F. Own Figure 5 — the complete OOD benchmark

She runs Original, A-LQR, S-PID, and H∞ herself under the same OOD conditions and decoding configuration.

- A: ID-to-OOD degradation for all methods, including Original.
- B: complete H∞ model-by-condition matrix.
- C: A-LQR versus H∞ as measured dynamics mismatch increases.
- D: H∞ advantage versus predicted fragility.

No values are imported from the user's benchmark runs. Her result unit is independently reproducible.

### G. Own Figure 6 — long-context stress test

She extends her own OOD runner to the frozen context lengths and produces:

- remaining target error;
- collateral utility retained;
- A-LQR error minus H∞ error.

She owns prompt construction, model execution, caching, evaluation, and the final figure for this experiment.

### H. Own Figure S2 — nonlinearity and certificate validity

She uses the same models and calibration pipeline from her robust-control track to produce:

- linearization residual versus distance from the nominal trajectory;
- predicted versus observed displacement;
- residual versus intervention strength and transformer depth;
- certificate validity versus distance from the nominal trajectory.

## User owns the baseline and residual track

The user owns only Figures 1, 2, and 4:

- Figure 1: conceptual framing.
- Figure 2: representation dynamics, residual checks, OOD/adversarial A-LQR failure exploration, and scale analysis.
- Figure 4: complete in-distribution Original/A-LQR/S-PID/H∞ benchmark, including the intervention-energy tradeoff.

The user also owns general dataset curation and any additional benchmark exploration not required by the collaborator's assigned figures.

## Independent compute and files

- Collaborator uses GPU 1 by default; user uses GPU 0.
- Collaborator uses her own cache and result directories so simultaneous runs cannot overwrite the user's work.
- Every collaborator experiment has its own configuration file, status file, and resumable output.
- She may use the same public datasets and existing repository helpers, but she does not wait for artifacts from the user.

## Recommended order for the collaborator

1. Figure S1 and H∞ correctness tests.
2. H∞ runtime and memory optimization.
3. One-model end-to-end smoke test using DistilGPT-2 or Qwen-2.5-0.5B.
4. Figure S3 gain and disturbance ablations on the small model.
5. Figure S2 nonlinearity and certificate validation.
6. Figure 3 robust-steerability prediction across models.
7. Figure 5 full OOD benchmark.
8. Figure 6 long-context stress test.

## Definition of done

Her track is complete when Figures 3, 5, 6, S1, S2, and S3 contain real values; their source units regenerate from cached or downloaded public inputs; H∞ passes numerical validation; and every baseline needed for those figures was run by her pipeline rather than supplied by the user.
