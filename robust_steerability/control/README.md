# Controller Contributor Guide

The controller package owns both offline controller synthesis and online
control behavior. It operates only on finite-horizon tensors; contributors do
not need to load a language model, register hooks, tokenize prompts, or work
through the experiment code.

## Shared Interface

Every controller subclasses `Controller` from `base.py` and exposes the same
online surface:

```python
controller.reset()
u = controller.control(layer_index, feedback_input)
delta = controller.intervention(layer_index, feedback_input)
```

- `control(k, feedback_input)` returns controller coordinates `u[k]`.
- `intervention(k, feedback_input)` returns the activation-space vector that
  the runtime applies. The base implementation computes `B[k] u[k]`; if a
  controller has no separate control channel, it returns `u[k]` directly.
- `reset()` clears online state before an independent rollout. The base
  implementation is a no-op for stateless controllers.

Controllers obtained from an offline control problem also subclass
`SynthesizedController` and implement:

```python
controller = ControllerClass.synthesize(problem, device="cuda:0", ...)
```

This class method runs offline synthesis and returns the same object used
online. CPU is suitable for small controller tests; main synthesis can run on
a selected GPU. The runtime never branches on controller names.

## File Ownership

- `base.py`: abstract interfaces and shared `B[k] u[k]` mapping.
- `types.py`: finite-horizon inputs and serializable synthesis results.
- `validation.py`: controller-boundary validation.
- `lqr.py`: LQR offline Riccati synthesis and online state feedback.
- `h_infinity.py`: H-infinity offline synthesis extension point and online
  state feedback.
- `pid.py`: stateful online PID behavior.
- `activation_addition.py`: fixed activation-addition baseline.
- `metrics.py`: controller-independent numerical evaluation.

Keep controller-specific offline and online mathematics in the controller's
own file. Shared structural behavior belongs in `base.py`. Model-facing state
definitions, such as semantic setpoints or full reference trajectories, are
adapted to controller feedback inputs in `runtime/policy.py`.

## Finite-Horizon Convention

The package uses:

```text
x[k + 1] = A[k] x[k] + B[k] u[k] + D[k] w[k]
u[k] = K[k] x[k]
```

The controller-ready problem contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `dynamics` | `(L, n, n)` | `A[k]`, nominal deviation dynamics |
| `control_channels` | `(L, n, m)` | `B[k]`, admissible intervention channels |
| `disturbance_channels` | `(L, n, r_w)` | `D[k]`, calibrated disturbance geometry |
| `state_costs` | `(L, n, n)` | stage state-performance costs |
| `control_costs` | `(L, m, m)` | stage intervention costs |
| `terminal_cost` | `(n, n)` | terminal state-performance cost |

Layer indices run from `0` through `L - 1`. A policy passes the state deviation
`x[k] = state[k] - reference[k]` to the controller. Returned gains therefore
follow the direct convention above, with no hidden sign change.

The optional `metadata` mapping records provenance such as model and tokenizer
revisions, state definition, coordinate normalization, behavior, calibration
split, and configuration hash. Controller mathematics must not branch on
model-specific metadata.

## Synthesis Output

Synthesized controllers can expose a `ControllerSolution` for serialization
and independent evaluation. It contains:

- gains shaped `(L, m, n)`;
- the controller name;
- feasibility status;
- `gamma_star` when applicable;
- numerical diagnostics useful for verification.

## H-Infinity Extension Point

Complete `HInfinityController.synthesize` in `h_infinity.py`. Its input is
already in deviation coordinates and contains `A`, `B`, `D`, stage costs, the
terminal cost, and gamma-search options. Implement the verified finite-horizon
recursion, saddle-point feasibility checks, and search for the smallest
feasible gamma, then construct the controller through `from_solution`.

The online path is already provisioned in the same class: `control` evaluates
the synthesized gain and `intervention` maps that control through `B`. If the
chosen H-infinity formulation requires additional online state, keep it in the
class and clear it in `reset`.

Do not add model loading, activation hooks, prompt handling, or experiment
logic to `h_infinity.py`. Do not silently substitute LQR when H-infinity
synthesis is unavailable or infeasible. Use small CPU fixtures and the
independent induced-gain evaluator for controller tests; full model experiments
remain integration checks owned by compact units.
