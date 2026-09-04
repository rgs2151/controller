# Controller Contributor Contract

Controller contributors work only with finite-horizon tensors. No knowledge of Hugging Face, transformer hooks, prompt datasets, or tokenization is required.

## Dynamics Convention

The package uses

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

All layer indices run from `0` through `L - 1`. The runtime applies returned gains directly with no hidden sign change.

The optional `metadata` mapping records provenance such as model and tokenizer revisions, state definition, coordinate normalization, behavior, calibration split, and configuration hash. Controller mathematics must not branch on model-specific metadata.

## Required Output

A solver returns `ControllerSolution` with:

- gains shaped `(L, m, n)`;
- the controller name;
- feasibility status;
- `gamma_star` when applicable;
- numerical diagnostics useful for verification.

## H-infinity Extension Point

Implement `synthesize_h_infinity` in `h_infinity.py`. The supplied problem is already expressed in deviation coordinates and includes the disturbance channel. The implementation must perform the verified finite-horizon recursion, enforce its saddle-point feasibility condition, and search for the smallest feasible gamma.

Do not add model-loading or activation-hook logic to a controller module. Do not silently substitute LQR when H-infinity synthesis is unavailable or infeasible.

Use the small CPU control fixtures and the independent induced-gain evaluator when validating the implementation. Full language-model experiments are integration checks owned by compact units, not controller unit tests.
