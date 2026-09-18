# Brain-as-LLM Local Linearization Experiment
## Analysis specification + figure-generation instructions

## 0. Scientific objective

We want to test whether human intracranial neural population dynamics can be treated, locally, in a way analogous to hidden-state dynamics in a large neural network.

The conceptual analogy is:

LLM:
    hidden state at layer l and token t
        h[l,t]

Brain:
    neural population state in anatomical module l at time t
        x[l,t]

Each anatomical region/module is treated as a "layer".
Each simultaneously recorded electrode channel within that region is treated as one unit of the population state.

The main questions are:

1. Can local neural dynamics be approximated by a linear model?

        x(t + Δ) ≈ A x(t)

2. Does the linear component capture the coarse neural trajectory while a learned residual correction improves prediction?

        x̂(t + Δ) = A x(t) + ŵ(t)

3. Over what temporal horizon does this local linearization remain useful?

4. Does a model learned in one neural/task state generalize to an altered state?

Primary OOD manipulation:

        train: placebo
        test-ID: held-out placebo
        test-OOD: scopolamine

A secondary OOD analysis can test cross-task transfer:

        free recall -> associative recall
        associative recall -> free recall

The final manuscript figure should communicate:

    locally linearizable
        ->
    finite temporal prediction horizon
        ->
    degradation under state/task shift


# 1. Important terminology

Do NOT call electrode contacts biological "neurons" in the manuscript.

Use:

- channels
- electrode channels
- neural units
- recording dimensions

The LLM analogy can describe each electrode as analogous to one hidden-state dimension, but we should not imply that one electrode corresponds to one biological neuron.


# 2. Dataset assumptions

Use the paper's encoding epochs.

Native recording:

    sampling rate = 1000 Hz

Trial epoch:

    -500 ms to +1500 ms relative to stimulus onset

Interpretation:

    -500 to 0 ms      = prestimulus baseline
       0 to 1500 ms   = encoding period

Do not mix recording channels from different patients into a simultaneous pseudo-population for the dynamical analysis.

Dynamics must be fit WITHIN PATIENT because electrodes from different patients were not simultaneously recorded.

Aggregate final statistics across patients.


# 3. Brain "layer" architecture

Use a conservative coarse anatomical hierarchy.

Primary architecture:

    Layer 1: lateral/temporal neocortex
        inferior temporal gyrus
        middle temporal gyrus
        superior temporal gyrus
        temporal-pole contacts where appropriate

                ↓

    Layer 2: entorhinal cortex

                ↓

    Layer 3: hippocampus

Anterior and posterior hippocampus should NOT automatically be treated as sequential layers.

Instead:

    hippocampus
        ├── anterior hippocampus
        └── posterior hippocampus

Treat anterior/posterior hippocampus as parallel regional subdivisions unless there is a specific anatomical reason to impose directionality.

Important:

The three-layer architecture is a coarse computational abstraction of the medial-temporal processing hierarchy. Do not describe every edge as proven monosynaptic feedforward connectivity.

For patients without simultaneous coverage of all three regions:

- perform within-region temporal modeling
- perform pairwise cross-region modeling where simultaneous coverage exists
- do not fabricate missing layers


# 4. State representation

## Primary analysis

The cleanest representation for the paper is band-limited instantaneous power.

For each electrode e and frequency band f:

    x_e,f(t) = baseline-normalized Hilbert power

Analyze frequency bands separately so that:

    one electrode = one state dimension

rather than concatenating all frequency bands and multiplying dimensionality.

Bands:

    slow theta: 2–4 Hz
    theta:      4–8 Hz
    alpha:      8–16 Hz
    beta:      16–32 Hz
    gamma:     32–64 Hz

Primary band for the initial demonstration:

    theta = 4–8 Hz

Also run:

    slow theta = 2–4 Hz

Then repeat all analyses over all five bands as a robustness analysis.


# 5. Temporal tokenization

Do NOT use each native 1-ms sample as a brain "token".

At 1 ms, adjacent neural measurements are highly autocorrelated and a linear model can appear artificially successful.

Primary token spacing:

    token_stride = 20 ms

Thus a 2-s epoch contains approximately:

    2000 / 20 = 100 temporal tokens

Conceptual analogy:

    LLM token index
        ↕

    brain time-bin index


## Feature estimation

Compute neural state at each 20-ms token.

Because Hilbert-power envelopes evolve more slowly than the carrier oscillation, 20-ms sampling of the envelope is acceptable.

Avoid future leakage.

Prefer a causal/local feature definition if the code permits it.

For example:

    x(t) = average power from [t - 40 ms, t]

rather than:

    [t - 20 ms, t + 20 ms]

The latter uses future data.

If standard zero-phase filtering/Hilbert processing is retained because this is an offline neuroscience analysis, explicitly document that fact and perform a causal-processing sensitivity analysis if practical.


# 6. Prediction horizons

The main temporal question is NOT just whether prediction works at one Δ.

Measure the linearization horizon.

Primary horizon:

    Δ = 20 ms

Full sweep:

    Δ ∈ {
        10 ms,
        20 ms,
        50 ms,
        100 ms,
        200 ms
    }

Optional finer sweep:

    5, 10, 20, 30, 50, 75, 100, 150, 200 ms

Do not choose the best horizon using the test set.


# 7. Analysis 1 — within-region linear dynamics

For each:

- patient
- anatomical region
- frequency band
- prediction horizon

construct

    x_t ∈ R^N

where N is the number of valid electrode channels in that region for that patient.

Fit:

    x_(t+Δ) = A x_t + b + ε_t

Estimate A and b using ridge regression.

Use regularization because:

- electrode counts may be moderately large
- channels are correlated
- the number of independent trials is limited

Choose ridge λ using TRAINING DATA ONLY.


# 8. Train / validation / test split

This is critical.

DO NOT randomly split individual time points.

Adjacent samples from the same trial are strongly correlated and would cause leakage.

Split by whole trials.

Example:

    60% training trials
    20% validation trials
    20% test trials

Prefer repeated group cross-validation when trial count permits.

Grouping variable:

    trial ID

If sessions exist:

An even stronger validation is leave-session-out or session-aware splitting.

All preprocessing parameters must be estimated from training data where appropriate:

- z-scoring
- PCA, if used
- ridge λ
- residual-model hyperparameters
- selected temporal lag


# 9. Baselines

Every linear model must beat trivial temporal baselines.

Calculate at least:

## Persistence

    x̂_(t+Δ) = x_t

## Mean temporal trajectory

For each training condition estimate:

    μ(t)

and predict:

    x̂_(t+Δ) = μ(t+Δ)

## Linear model

    x̂_(t+Δ) = A x_t + b

Optional:

## Diagonal AR model

Each electrode predicts only itself.

This asks whether the full A matrix gains anything from population coupling.


# 10. Analysis 2 — linear component + residual correction

This analysis corresponds to the figure showing:

    black = ground truth
    blue  = Ax
    red   = Ax + residual correction


## DO NOT DO THIS

Do not calculate the test residual

    w_t = x_(t+Δ) - A x_t

and then add it back to the prediction.

That would trivially reproduce the target and is not a prediction.


## Correct procedure

First fit the linear model on training data:

    x̂_linear = A x_t + b

Define TRAINING residuals:

    r_t = x_(t+Δ) - x̂_linear


Then learn a separate residual predictor:

    ŵ_t = g(x_t)

Possible g models, in preferred order:

### Option A — low-capacity MLP

    input: x_t
    hidden layer: small
    output: predicted residual vector

Keep this deliberately small.

Example:

    N -> min(32, 2N) -> N

Use strong regularization and early stopping.


### Option B — kernel ridge

Useful if N is small.


### Option C — linear residual with task context

If task covariates u_t are available:

    ŵ_t = B u_t

giving

    x̂ = A x_t + B u_t


Final prediction:

    x̂_(t+Δ) = A x_t + b + ŵ_t


Interpretation:

    A x_t
        = coarse locally linear component

    ŵ_t
        = predictable nonlinear/context-dependent correction


# 11. Linearization-error metric

For every sample calculate:

    e_linear =
        ||x_(t+Δ) - A x_t||_2
        ---------------------
        ||x_(t+Δ)||_2 + ε

Also calculate residual-energy fraction:

    E_residual =
        ||x_(t+Δ) - A x_t||²
        ----------------------
        ||x_(t+Δ)||²

Plot these quantities as a function of Δ.

This is a direct estimate of where the linear approximation stops being locally useful.


# 12. Main predictive metrics

Report:

1. Population R²

2. Normalized RMSE

       NRMSE = RMSE / SD(test target)

3. Pearson correlation between true and predicted trajectory

4. Cosine similarity of population states

       cos(x_true, x_pred)

5. Relative residual energy

6. Improvement from learned residual

       ΔR² =
           R²(Ax + ŵ) - R²(Ax)

Primary manuscript metric:

    cross-validated R²

Do not rely only on correlation because correlation can remain high despite large amplitude errors.


# 13. Analysis 3 — temporal linearization horizon

For every model calculate performance at:

    10
    20
    50
    100
    200 ms

Expected qualitative pattern:

prediction
quality
  |
  |  ●
  |    ●
  |       ●
  |            ●
  |                    ●
  +---------------------------
     10 20 50 100 200 ms

Do not force monotonicity in the code.

Measure it from data.


Fit separately:

    A-only
    A + residual predictor
    persistence baseline


The key claim is supported if:

1. Ax significantly predicts future neural state above trivial baselines.

2. Ax + ŵ improves prediction.

3. Performance declines with increasing Δ.

This would support a finite local linearization horizon.


# 14. Analysis 4 — brain state OOD test

Primary OOD test:

TRAIN MODEL USING PLACEBO ONLY.

Then evaluate WITHOUT REFITTING:

## ID

    held-out placebo trials

## OOD

    scopolamine trials


Important:

The exact same:

    A
    b
    residual model g
    normalization
    hyperparameters

must be applied to the OOD condition.


Calculate:

    R²_ID
    R²_OOD

and

    OOD degradation =
        R²_ID - R²_OOD


Also calculate:

    residual_energy_ID
    residual_energy_OOD


Strong prediction:

    residual energy increases under OOD

and/or

    R² decreases under OOD.


# 15. Symmetric OOD control

Also reverse the experiment:

    train scopolamine
    test scopolamine   = ID
    test placebo       = OOD

This determines whether the effect reflects:

- general state mismatch

rather than simply:

- scopolamine signals being inherently harder to predict.


# 16. Secondary OOD test — task transfer

If free-recall and associative-recall data are sufficiently comparable:

Experiment A:

    train FR
    test FR
    test AR

Experiment B:

    train AR
    test AR
    test FR

Treat this as secondary because the task structure itself differs.

Drug-state transfer is the cleaner primary OOD experiment.


# 17. Analysis 5 — cross-region "layer" mapping

After the within-region temporal model is established, test the LLM-layer analogy more directly.

For adjacent anatomical modules:

    Temporal cortex -> Entorhinal cortex

and

    Entorhinal cortex -> Hippocampus


Fit:

    x^(l+1)_(t+δ)
        =
    A_l x^l_t + b


Potential interregional lags:

    δ = 0
        10
        20
        50
        100 ms

Select δ using TRAINING DATA ONLY.

Report test performance at every δ rather than only the optimum.


Optional model accounting for the target region's own current state:

    x^(l+1)_(t+δ)
        =
    A_l x^l_t
        +
    R_l x^(l+1)_t
        +
    b

This is scientifically more conservative because future activity in a region depends both on:

- incoming population activity
- its own recurrent state


Compare:

    recurrent-only:
        R_l x^(l+1)_t

vs.

    source + recurrent:
        A_l x^l_t + R_l x^(l+1)_t


The gain from A_l quantifies additional predictive information from the upstream module.


# 18. Matrix diagnostics

For every fitted A calculate:

- eigenvalue spectrum
- spectral radius
- singular values
- effective rank
- condition number

Potentially useful visualization:

    eigenvalues of A in the complex plane

This could connect the brain analysis more explicitly to dynamical-systems/control language.

Do not overinterpret eigenmodes if model stability is poor.


# 19. Statistical unit

The inferential unit should be:

    patient

not:

    individual time bin
    individual electrode
    individual trial-time sample


For every patient obtain one summary metric for the relevant comparison.

Then perform paired statistics across patients.

Example:

    R² linear
        vs
    R² linear + residual

and:

    ID
        vs
    OOD


Show individual patient dots whenever possible.


# 20. Important controls

Run the following sanity checks.

## Temporal shuffle

Randomize temporal correspondence between x_t and x_(t+Δ).

Prediction should collapse.


## Trial shuffle

Pair source states and future states from different trials.

Prediction should strongly decrease.


## Persistence comparison

Ax must outperform x_t -> x_(t+Δ) persistence.


## Electrode-count control

Performance may vary with N.

Repeat analyses after subsampling equal numbers of electrodes where needed.


## Frequency-band analysis

Repeat across:

    slow theta
    theta
    alpha
    beta
    gamma


## Patient-wise robustness

No single patient should drive the central claim.


# 21. Output files the analysis script should generate

Create machine-readable intermediate outputs.

## Tables

    results/
        model_metrics.csv
        horizon_metrics.csv
        ood_metrics.csv
        layer_metrics.csv
        patient_metadata.csv
        electrode_metadata.csv

Each row should include:

    patient
    region
    task
    drug condition
    frequency band
    Δ
    model type
    fold
    R²
    RMSE
    correlation
    cosine similarity
    residual energy


## Saved predictions

    predictions/
        patient_X_region_Y_band_Z.npz

Include:

    time
    true
    prediction_linear
    prediction_linear_residual
    trial ID
    condition


# 22. MAIN FIGURE DESIGN

Create one publication-quality figure with four conceptual panels.

Recommended overall structure:

┌──────────────────────────┬─────────────────────────────────────┐
│ A                        │ B                                   │
│ Brain-as-network mapping │ Local linearization                 │
│ + task + neural signals  │ + representative prediction        │
├──────────────────────────┼─────────────────────────────────────┤
│ C                        │ D                                   │
│ Temporal horizon         │ OOD/state-shift failure             │
│ + quantitative results   │ + quantitative comparison          │
└──────────────────────────┴─────────────────────────────────────┘


# PANEL A — Neural architecture / brain as network

Title:

    Neural population dynamics as a layered system


Top:

show schematic anatomical hierarchy:

    Temporal cortex
        ->
    Entorhinal cortex
        ->
    Hippocampus


Under each module show several small electrode traces.

Represent electrode channels as small circles / vertical nodes.

Label:

    x^(1)_t
    x^(2)_t
    x^(3)_t


Add a small conceptual analogy:

    anatomical module ≈ network layer
    electrode channel ≈ state dimension
    time bin ≈ token


Bottom of Panel A:

show task-aligned temporal window:

    -500 ms                 0                    1500 ms
      |---------------------|------------------------|
           baseline          encoding

Overlay 20-ms token marks schematically.

Do not draw all 100 bins.

Only visually indicate discrete sampling.


# PANEL B — Local linearization

Title:

    Neural dynamics are locally linearizable


Top half:

show mathematical block:

                ┌── A ──┐
    x_t  ------>|       |------> A x_t
                └───────┘
                              \
                               + ----> x̂_(t+Δ)
                              /
                         ŵ_t


Equation:

    x̂_(t+Δ) = A x_t + ŵ_t


Use labels:

    A x_t
        linear / coarse dynamics

    ŵ_t
        learned residual correction


Bottom:

representative real test trajectory.

Use three curves:

    BLACK:
        true neural trajectory

    BLUE:
        linear prediction A x_t

    RED:
        linear + predicted residual


The blue curve should capture the gross temporal pattern.

The red curve should be closer to black.

Do NOT use synthetic traces in the final figure.

Select one representative patient/region using a PREDEFINED criterion, e.g. patient closest to median improvement.

Do not cherry-pick the visually best patient.


# PANEL C — Finite temporal horizon

Title:

    Linearization has a finite prediction horizon


Main graph:

x-axis:

    prediction horizon Δ (ms)

        10
        20
        50
        100
        200

y-axis:

    cross-validated R²


Plot:

    persistence baseline
    A x_t
    A x_t + ŵ_t

Use patient-averaged curves with confidence intervals.

Overlay faint patient-level values if visually feasible.


Expected visual story:

At short Δ:

    linear model works

At intermediate Δ:

    residual model provides additional gain

At longer Δ:

    both models deteriorate


Optional inset:

    relative residual energy vs Δ


Add a subtle annotation near Δ=20 ms:

    primary analysis


# PANEL D — OOD neural-state shift

Title:

    Local model degrades under neural-state shift


Top:

schematic:

    TRAIN
        placebo

        ↓ frozen model

        ├── placebo test
        │       ID
        │
        └── scopolamine test
                OOD


Bottom left:

representative prediction trace under OOD.

Same plotting convention:

    black = ground truth
    blue  = Ax
    red   = Ax + ŵ


Bottom right:

paired patient plot or bar + patient dots:

            ID placebo       OOD scopolamine

    Linear
        ●                   ●

    Linear + residual
        ●                   ●


Primary y-axis:

    R²

Optional second metric:

    relative residual energy


The strongest visual message should be:

    same frozen model
          +
    different neural state
          ->
    increased linearization error


# 23. Visual style

Use restrained publication-style graphics.

General:

- white background
- no 3D bar plots
- minimal borders
- consistent typography
- vector output
- axes only where necessary
- avoid decorative neuroscience graphics that imply unsupported anatomy

Core prediction colors should remain identical across panels:

    ground truth            = black
    linear prediction Ax    = blue
    Ax + residual           = red

Use a separate muted encoding for experimental condition:

    placebo / ID
    scopolamine / OOD

Do not change the meaning of blue/red between panels.

Panel labels:

    A B C D

large bold letters in upper-left corners.


# 24. Figure narrative

A reader should understand the entire argument without reading the Results section.

Panel A asks:

    What is the neural state and what corresponds to an LLM layer?

Panel B asks:

    Can the next neural state be locally predicted by a linear map?

Panel C asks:

    For how long is this approximation valid?

Panel D asks:

    Does that approximation survive a change in neural state?


The visual progression should therefore be:

    REPRESENTATION
          ↓
    LINEARIZATION
          ↓
    TEMPORAL LIMIT
          ↓
    DISTRIBUTION SHIFT


# 25. Claims we are allowed to make if the results support them

Preferred cautious wording:

    "Population dynamics were locally approximated by a linear
    state-transition model over short temporal horizons."

    "A learned residual component explained additional predictable
    structure not captured by the linear transition."

    "Predictive performance decreased with increasing temporal
    horizon, indicating a finite regime of useful local linearization."

    "Models fit in the placebo state generalized less accurately
    to scopolamine activity, indicating state-dependent changes
    in the local dynamical approximation."


Avoid:

    "The brain is linear."

    "Brain regions are transformer layers."

    "Scopolamine makes the brain nonlinear."

    "Residuals prove nonlinear computation."

The experiment measures the quality and transferability of a local approximation.


# 26. Suggested analysis order

Implement in this order:

STEP 1
    Load / validate electrode localization and trial metadata.

STEP 2
    Reproduce paper-style preprocessing and band-power signals.

STEP 3
    Build patient-specific region state vectors.

STEP 4
    Implement trial-wise train/validation/test splitting.

STEP 5
    Fit persistence baseline.

STEP 6
    Fit ridge linear model A.

STEP 7
    Generate horizon curve.

STEP 8
    Fit learned residual model.

STEP 9
    Compare Ax vs Ax + ŵ.

STEP 10
    Run placebo -> placebo/scopolamine OOD analysis.

STEP 11
    Run symmetric scopolamine -> scopolamine/placebo control.

STEP 12
    Run frequency-band robustness.

STEP 13
    Run cross-region layer analysis.

STEP 14
    Run temporal/trial shuffle controls.

STEP 15
    Generate patient-level statistics.

STEP 16
    Create the four-panel manuscript figure.


# 27. First implementation target

Before implementing every analysis, create one end-to-end proof of concept:

Patient:
    choose one subject with sufficient electrodes

Region:
    hippocampus

Band:
    theta 4–8 Hz

Time:
    0–1500 ms post-stimulus

Token stride:
    20 ms

Prediction horizon:
    20 ms

Models:

    persistence

    x_(t+20ms) = A x_t

    x_(t+20ms) = A x_t + ŵ_t


Output:

1. one true-vs-predicted trajectory
2. R² for the three models
3. residual-energy ratio
4. placebo-ID vs scopolamine-OOD comparison


Only after this passes sanity checks should the complete batch analysis be run.


# 28. Final implementation principle

Prioritize scientific validity over making the desired story work.

If:

    A does not beat persistence

report that.

If:

    residual correction does not improve held-out prediction

report that.

If:

    OOD degradation is absent

report that.

The purpose of the analysis is to measure the regime in which a
local linear approximation is useful, not to force the brain data to
behave like an LLM.
