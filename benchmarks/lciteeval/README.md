# L-CiteEval benchmark

This first-class benchmark runs the frozen AXBench concept-499 direction through the L-CiteEval-Length HotpotQA transfer conditions.

- `artifacts`: materialize the 72/72 AXBench direction data, fit layer-wise all-token DiffMean setpoints, and average 50 desired-example Jacobians into the one A matrix shared by A-LQR and H∞.
- `calibrate`: write fixed S-PID/A-LQR selections and select H∞ on all 40
  HotpotQA 8K prompts using 45% answer recall, 45% citation F1, 5% AXBench
  concept relevance, and 5% AXBench fluency. This intentionally tunes on the
  matched question identities later evaluated at longer context lengths.
- `evaluate`: generate one deterministic answer for each of 40 matched questions at 8K and 16K with Original, S-PID, A-LQR, and H∞.
- `score`: independently compute L-CiteEval answer overlap, citation NLI, and the AXBench steering scores from saved generations.

The 8K and 16K conditions share the same 40 question identities and every controller artifact. The registered 32K condition is deferred and can be added later with `--datasets 32k`. Context length changes only at evaluation; no direction, A matrix, disturbance model, gain, or setpoint is refit by length.

The 50 nominal Jacobians use the upstream concept-pipeline limit of 32 tokens.
The target model is loaded in bfloat16 without quantization; Qwen uses the same
static YaRN configuration at every evaluation length. H∞ calibration records
the exact shared A-matrix signature and AXBench direction identity, and runtime
refuses a controller synthesized against different shared artifacts.

## Persisted outputs

Each model has one cache rooted at
`~/robust-steering-cache/lciteeval/<model>/` on Lightning and
`benchmarks/lciteeval/cache/<model>/` locally:

- `artifacts/` contains the materialized direction data, setpoint, and the
  averaged A matrix. Individual Jacobians are reduced to resumable running
  sums during fitting and removed after the final average is saved.
- `calibrations/<method>/<calibration-id>/` contains every method selection.
  The H∞ directory additionally contains disturbance residuals, all grid
  controllers/generations/scores, the selected controller, and Hannah's exact
  portable diagnostic run: `calibration_input.pt`, `controller.pt`,
  `score.json`, `manifest.json`, `controller_source.py`, and
  `exporter_source.py`.
- `evaluations/<kv-cache-condition>/` contains resumable generation shards,
  merged generations, prompt-level scorer outputs, and summaries. Generation
  shards are written after every batch and citation scores after every answer.
- Named recalibrations preserve the historical `selected` run and write their
  complete evaluation state below
  `evaluations/<kv-cache-condition>/calibrations/<calibration-id>/`.
- `benchmarks/lciteeval/results/<kv-cache-condition>/` contains only compact
  Git-tracked summaries. Named recalibrations use the matching
  `calibrations/<calibration-id>/` subtree; large tensors, prompts, and
  generations remain in the ignored persistent cache.

```bash
python -m robust_steerability.benchmarks.lciteeval artifacts --model qwen25_3b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval calibrate --model qwen25_3b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval evaluate --model qwen25_3b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval score --model qwen25_3b_instruct --scorers default --devices auto
```
