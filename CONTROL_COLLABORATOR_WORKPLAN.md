# Control-Theory Collaborator Workplan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-07
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Frozen reference

The visual specification in `figs/sketch/` is frozen at commit `415c45b`. It is read-only. Real analyses and final plots should be built in new result units; they should not overwrite the sketch or its placeholder values.

## Division principle

The control collaborator owns everything that can be completed from numerical state-space arrays. The LLM/benchmark owner owns everything that requires model loading, tokenization, prompt design, activation hooks, text generation, or behavioral scoring.

The boundary is already represented by:

- `FiniteHorizonControlProblem`: A, B, D, Q, R, and terminal-cost tensors in.
- `ControllerSolution`: gains, feasibility, gamma-star, and diagnostics out.
- `Controller.intervention(...)`: the online numerical intervention supplied to the model runtime.

The collaborator should not need to inspect prompts or understand Hugging Face internals.

## Control collaborator: primary ownership

### 1. Verify the H∞ implementation mathematically

- Audit the finite-horizon minimax recursion, feedback sign, terminal cost, and feasibility condition in `robust_steerability/control/h_infinity.py`.
- Verify scalar, MIMO, rectangular-control, rectangular-disturbance, and time-varying systems.
- Check that feasibility is monotone in candidate gamma and that reported gamma-star is the smallest feasible value within the stated tolerance.
- Compare synthesized results against the explicit induced-gain calculation in `robust_steerability/control/metrics.py` for small systems.
- Add tests for nearly singular saddle-point matrices, infeasible search intervals, dtype conversion, serialization, and CPU/GPU agreement.

Acceptance: every feasible small-system solution satisfies the independently computed disturbance-gain bound within numerical tolerance, and failure cases return explicit diagnostics.

### 2. Optimize H∞ synthesis

- Benchmark runtime and peak memory across horizon, state dimension, control rank, disturbance rank, CPU, and one GPU.
- Remove repeated device transfers and allocations across gamma-search iterations.
- Reuse factorization or workspace where mathematically valid.
- Evaluate stable factorization choices near the feasibility boundary.
- Preserve float64 synthesis accuracy while documenting when float32 online gains are safe.
- Add performance regression benchmarks and a concise runtime/memory report.

Acceptance: numerical answers remain within tolerance while runtime and peak-memory changes are measured, not guessed.

### 3. Complete controller-side artifacts and diagnostics

- Make controller problems and solutions reproducibly serializable with configuration metadata.
- Export gamma-star, S_rob = 1/gamma-star, feasibility bracket, margins, condition numbers, and failure layer/reason.
- Add controller-only trajectory simulation and stage-wise target cost, intervention energy, collateral cost, and total performance energy.
- Keep all of this independent of Transformers and datasets.

Acceptance: a saved numerical problem can be synthesized, restored, simulated, and summarized without loading an LLM.

### 4. Own Figure S1: H∞ validation

Create a real result unit for the four frozen panels:

- A: feasibility margin versus candidate gamma, with gamma-star marked.
- B: synthesized gamma-star versus an independent exact finite-horizon result.
- C: performance-output energy versus disturbance energy and the certified bound.
- D: observed tracking error versus certified tracking-error bound.

This figure is entirely controller-side and can be completed without any LLM work.

### 5. Own the controller machinery for Figure S3

- Panels A–B: implement the tuning sweep and return target-cost, collateral-cost, intervention-energy, feasibility, and margin tables.
- Do not scale an H∞ gain after synthesis if that invalidates its certificate. Decide the mathematically valid sweep—such as resynthesizing across Q/R/S weights or another documented controller parameter.
- Panel C: consume problem bundles created at different calibration sizes and report gamma-star stability, feasibility rate, and downstream prediction-ready summaries.
- Panel D: construct energy-matched disturbance geometries: isotropic, learned residual, target-aligned, and worst-case/adversarial directions. Return controller and certificate summaries for each.

The LLM owner later supplies the empirical steering-success and utility columns. The collaborator then renders the final S3 plots from the merged table.

### 6. Own controller-side quantities used in the main figures

- Figure 3: compute gamma-star, S_rob, feasibility diagnostics, and controller-derived competing predictors for every supplied model–behavior problem.
- Figure 4D: compute normalized intervention energy and controller cost from recorded control sequences.
- Figure 5C–D: compute dynamics-mismatch summaries, fragility, and the H∞-versus-A-LQR comparison fields after empirical reliability is supplied.
- Figure S2: compute certificate validity and theoretical/empirical bound comparisons from supplied trajectories.

The collaborator may own plotting and statistical code after receiving tidy result tables, but she does not generate the LLM observations in those tables.

## LLM/benchmark owner: primary ownership

- Choose models, behaviors, prompt datasets, ID/OOD/adversarial conditions, and train/calibration/test splits.
- Load models and tokenizers, register activation hooks, collect hidden states and Jacobians, and fit semantic targets.
- Produce residual tensors and numerical A/B/D/Q/R/terminal-cost problem bundles for the collaborator.
- Integrate returned controller solutions into the Hugging Face runtime.
- Run GPU generations for Original, A-LQR, S-PID, and H∞ under identical decoding settings.
- Measure truthfulness, toxicity, informativeness, utility retention, and other behavioral outcomes.
- Own Figures 1, 2, 4A–C, 5A–B, and 6.

One current integration issue belongs on this side: `AppliedControler/run_steering.py` presently sets the H∞ disturbance channel equal to the identity control channel. The final pipeline must instead pass the calibration-derived disturbance geometry supplied through the agreed numerical boundary.

## Shared handoff artifacts

### LLM owner to collaborator

For each model–behavior pair and calibration replicate:

- unique problem ID and split/configuration hash;
- A, B, D, Q, R, and terminal-cost tensors;
- calibration size and disturbance-geometry label;
- optional held-out residual sequences for certificate evaluation;
- no prompt text is required.

### Collaborator to LLM owner

- serialized `ControllerSolution` for each requested configuration;
- gamma-star and S_rob;
- feasibility and numerical diagnostics;
- controller tuning/sweep table;
- predicted control, energy, collateral, and certificate summaries;
- exact code/configuration hash used to produce them.

### LLM owner back to collaborator

- tidy evaluation table containing problem ID, method, condition, seed, empirical steering reliability, intervention energy, and collateral/utility metrics.

The collaborator can then populate Figure 3, Figure 4D, Figure 5C–D, Figure S2, and Figure S3 without opening the model pipeline.

## Parallel execution order

1. Collaborator immediately completes H∞ verification, optimization, controller metrics, and Figure S1 using synthetic numerical systems.
2. In parallel, the LLM owner exports real numerical problem bundles and finishes the benchmark runner.
3. Collaborator consumes those bundles, produces controller solutions, gamma-star values, and S3 sweep/ablation tables.
4. LLM owner runs the returned solutions on the GPUs and produces the empirical outcome table.
5. Collaborator fills the controller-heavy plots from that table; the LLM owner fills the remaining benchmark panels.

## Leakage and comparison rules

- Gamma-star and S_rob use calibration data only.
- Controller tuning cannot use the held-out OOD/adversarial test outcomes later used for the headline claims.
- A-LQR and H∞ must be compared at matched intervention energy or matched collateral degradation.
- Disturbance geometries must be energy-normalized before comparison.
- Prompt-level replicates cannot be treated as independent model–behavior pairs in the prospective-prediction analysis.
