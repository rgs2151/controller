# Integration Checklist

## Goal

Apply three steering methods on the same model/benchmark slices:

- Method A: Standard baseline method from your provided source
- Method B: Standard baseline method from your provided source
- Method C: New method to develop

## Experimental Scope (Current Default)

- Models: 10 merged benchmark models (including Qwen 72B)
- Primary targets: `selected_ood_targets_top1.csv`
- Stress-test targets: `selected_ood_targets_top3.csv`

## Build Plan

1. Define a common controller interface adapter. ✅ (`method_registry.py`)
2. Wrap existing repo controllers (A-LQR and PID) into that interface. ✅ (`alqr`, `spid`, plus `actadd`)
3. Implement Method A from provided reference.
4. Implement Method B from provided reference.
5. Implement Method C (new method).
6. Add a single evaluation runner with consistent decode and metric settings.
7. Export per-method comparison tables/plots.

## Scaffold Outputs

- `method_registry.py`: pluggable method adapters (built-in + deferred placeholders).
- `plan_experiments.py`: emits model/subset/method experiment plans from top1/top3 CSV targets.

## Reproducibility Requirements To Capture Per Run

- Model name and revision
- Benchmark subset name
- Prompt count and prompt seed
- Max length and decoding config
- Controller/method name and hyperparameters
- Any fitted offline artifacts and file hashes
- Runtime environment details (GPU, dtype/quantization)

## Ready-Now Decision

The next concrete step is to receive the method repository links and your new method formula/spec and then scaffold the first runner script.
