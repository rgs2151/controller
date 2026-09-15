# Truthfulness benchmark

This unit owns the complete TruthfulQA pipeline and its Spanish transfer test. It is portable across the local workstation and directly accessed Lightning Studios.

## Stages

1. `artifacts` freezes the 200 false and 200 true fit records, fits the semantic setpoint, and computes the exact average of 35 independent desired-prompt Jacobians.
2. `calibrate` records controller choices for the model-task pair. A-LQR adopts the published setting; evaluated methods must have an explicit selected calibration.
3. `evaluate` runs five full 817-question repetitions and scores True, Informative, their product, AXBench instruction relevance, and AXBench fluency. Spanish uses the matched translated questions, English answers, and the identical judges.

Evaluation never silently runs calibration. KV cache is off unless an explicit cache-on run is requested.
The two AXBench measures use the published independent 0--2 rubric definitions
with `gpt-4o-mini` at temperature zero. For throughput, one strict structured
request scores batches of 20 responses on both rubrics; the two scores remain
separate. The `quality` stage scores existing cached generations without
regenerating model responses or rerunning the TruthfulQA judges.

## Commands

```bash
python -m robust_steerability.benchmarks.truthfulness artifacts --model llama8b --devices auto
python -m robust_steerability.benchmarks.truthfulness calibrate --model llama8b --methods original,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.truthfulness evaluate --model llama8b --methods original,alqr,h_infinity --datasets id,spanish --kv-cache off --devices auto
python -m robust_steerability.benchmarks.truthfulness quality --model llama8b --methods original,alqr,h_infinity --datasets id,spanish --kv-cache off
```

On a larger GPU, append `--generation-batch-size <n>` to `calibrate` or
`evaluate`. The selected value is part of cache identity and the tracked run
record; omitting it uses the pinned model/method default.

## Ownership

```text
cache/<model>/artifacts/
cache/<model>/calibrations/<method>/<calibration-id>/
cache/<model>/evaluations/<kv-cache-condition>/
results/<kv-cache-condition>/<model>/<dataset>/<method>.json
```

`cache/` is machine-local and ignored by Git. `results/` and the corresponding root `logs/<run-id>/` records are tracked. Publish only reusable large artifacts and calibrations to S3 with `python -m robust_steerability.storage.s3`; model weights stay in that machine's Hugging Face cache.
OpenAI quality-judge requests require `OPENAI_API_KEY` in the environment or in
the checkout's ignored `.env` file.
