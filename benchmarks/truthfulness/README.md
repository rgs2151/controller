# Truthfulness benchmark

The benchmark has four explicit stages. Each stage can run independently on the machine that owns the model's local cache.

1. `artifacts` freezes the 200 false examples, 200 true examples, semantic setpoint, and average of 35 desired-prompt Jacobians.
2. `calibrate` records method parameters. A-LQR adopts its published values; H∞ selects its Q/R and Qf/R grid point by mean True percentage on five repetitions of the fixed tuning set.
3. `evaluate` generates five complete 817-question repetitions for TruthfulQA ID and/or matched Spanish questions. Evaluated-model KV cache is always disabled. No judge runs here.
4. `score` applies any requested subset of independent judges to cached responses and writes a summary from every completed judge cache.

The truthfulness score stage supports:

- `true`: pinned TruthfulQA True judge, binary score reported as a percentage.
- `informative`: pinned TruthfulQA Helpful judge, binary score reported as a percentage.
- `instruction_relevance`: AXBench 0–2 instruction-relevance rubric through `gpt-4o-mini`.
- `fluency`: AXBench 0–2 fluency rubric through `gpt-4o-mini`.

Concept relevance remains available in the shared `robust_steerability.judges` package for benchmarks that supply an explicit target concept. It is not a TruthfulQA metric.

## Commands

```bash
python -m robust_steerability.benchmarks.truthfulness artifacts --model llama8b --devices auto
python -m robust_steerability.benchmarks.truthfulness calibrate --model llama8b --methods alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.truthfulness evaluate --model llama8b --methods original,alqr,h_infinity --datasets id,spanish --devices auto
python -m robust_steerability.benchmarks.truthfulness score --model llama8b --methods original,alqr,h_infinity --datasets id,spanish --judges true,informative,instruction_relevance,fluency --devices auto
```

`--generation-batch-size` changes only evaluated-model throughput. `--api-concurrency` defaults to 500 and `--api-batch-size` defaults to 20 for the API judges. OpenAI scoring reads `OPENAI_API_KEY` from the environment or the checkout's ignored `.env` file.

## Ownership

```text
cache/<model>/artifacts/
cache/<model>/calibrations/<method>/<calibration-id>/
cache/<model>/evaluations/kv_cache_off/generations/
cache/<model>/evaluations/kv_cache_off/judges/<judge>/
cache/<model>/evaluations/kv_cache_off/results/
results/kv_cache_off/<model>/<dataset>/<method>.json
```

Large artifacts and calibrations may be published to S3. Generations, independent judge caches, summaries, and run logs remain owned by the machine that ran them until the tracked summaries and logs are committed.
