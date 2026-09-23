## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-22T03:30:33.059473+00:00
- Verification Status: ANALYZED
- Version Label: exp_result_v1

## Experiment Result

- **ID**: computational-complexity-llama32-1b-harmful-20260922
- **Type**: analysis
- **Status**: completed
- **Command**: `conda run -n robust-steerability python parking/computational_complexity/computational_complexity.py --device cuda:0 --batch-size 8 --synthesis-repeats 20 --inference-repeats 7 --bootstrap-refits 100 --recompute`
- **Working Directory**: `/home/dev/controller/code`
- **Duration**: 141.684 seconds
- **Exit Code**: 0

### Output Files

| File | Purpose |
|---|---|
| `cache/run_manifest.json` | Exact execution parameters and timestamps |
| `cache/hardware_software.json` | Hardware, software, model, and source provenance |
| `cache/synthesis_trials.csv` | 60 raw synthesis measurements |
| `cache/gamma_trace.csv` | 24 production bisection states |
| `cache/bisection_bootstrap_refits.csv` | 100 calibration bootstrap fits |
| `cache/inference_trials.csv` | 21 raw generation measurements |
| `cache/summary.json` | Frozen numerical summary |
| `plots/computational_complexity.pdf` | Main four-panel appendix figure |
| `plots/bisection_stability.pdf` | Bootstrap gamma-star stability figure |
| `plots/complexity_table.pdf` | Standalone summary table |
| `plots/appendix_evidence.md` | Appendix M evidence outline |
| `plots/reviewer_defense_matrix.md` | Reviewer-critique mapping |
| `plots/reviewer_defense_matrix.pdf` | Standalone reviewer-defense table |

### Output Summary

- H-infinity synthesis median: 0.124515 s; production gamma-star: 0.13007522; production and all 100 bootstrap refits: 24 bisection iterations.
- Median latency: Original 5.347, A-LQR 5.409, and H-infinity 5.459 ms/token.
- Median peak allocated VRAM: 2.882 GB for all three methods.

### Anomalies Detected

- The supplied plan describes bisection iterations across prompts, but this implementation synthesizes one prompt-aggregated controller. The unit reports the exact production iteration count and 100 bootstrap refits of calibration disturbance geometry instead.
- The supplied plan proposes “identical” A-LQR/H-infinity deployment complexity. The deployed policies use different coordinate systems and measured latency is close but nonidentical; the unit preserves that distinction.
