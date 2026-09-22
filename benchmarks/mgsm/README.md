# MGSM multilingual transfer benchmark

This benchmark constructs one English-to-Spanish residual direction from all 250
matched MGSM test-question pairs, then evaluates a fixed 100-problem subset in
five transfer languages: Chinese, French, Japanese, Swahili, and Telugu.

- `artifacts`: materialize the paired direction corpus, fit the per-layer DiffMean
  setpoint, and average 50 Spanish-question Jacobians into the nominal `A` shared
  by A-LQR and H∞.
- `calibrate`: write the frozen A-LQR selection and run the 12-point H∞
  Q/R–Qf/R grid at the same fixed `lambda=1.5` target used by A-LQR. The optional
  lambda-sweep machinery remains available but is disabled. H∞ fits its
  disturbance geometry on 200 translated GSM8K prompts balanced across Bengali,
  German, Russian, and Thai, then selects the cost grid on 100 additional disjoint
  prompts in the same four-language mixture. Selection uses the mean per-response
  accuracy-weighted additive combination of 90% exact-answer accuracy and 10%
  normalized AXBench Overall.
- `evaluate`: generate one native-eight-shot worked solution for each of 100
  matched problems in every requested input language, capped at 256 new tokens.
- `score`: independently compute exact final-number accuracy, AXBench Spanish rule
  following, instruction relevance, fluency, and their steering harmonic mean.
  Exact-number scoring accepts decimal and thousands separators in either comma
  or period conventions and resolves ambiguous final tokens against the numeric
  gold answer.

English and Spanish construct the direction and are not in the default evaluation.
The remaining registered languages stay independently runnable with `--datasets`;
evaluation never refits the controller by language.

Qwen3-4B is complete under its original English calibration protocol. The first
Llama-3.2-3B-Instruct H∞ run exposed calibration saturation and is retained for
comparison. Phi-4-mini-instruct and Granite-3.3-2B-Instruct are registered as
three-method model expansions and use the four-language calibration specified
above. S-PID remains available as an optional composable method, but is not part
of the active MGSM protocol.

```bash
python -m robust_steerability.benchmarks.mgsm artifacts --model llama32_3b_instruct --methods original,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm calibrate --model llama32_3b_instruct --methods original,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm evaluate --model llama32_3b_instruct --methods original,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm score --model llama32_3b_instruct --methods original,alqr,h_infinity --scorers default --devices auto
```

For the Phi or Granite expansion, replace the model key above with
`phi4_mini_instruct` or `granite33_2b_instruct`. Each model's artifacts,
calibration, generations, scores, and results remain isolated under its model
key.

Use `--datasets all` only for the deferred 11-language expansion.

## Fixed A-LQR reruns

An A-LQR-only rerun reuses the saved MGSM setpoint and nominal dynamics; it does
not refit the direction, Jacobians, or shared dynamics matrix. Write the explicit
fixed selection first, then replace only the requested A-LQR generations and
their scorer outputs:

```bash
python -m robust_steerability.benchmarks.mgsm calibrate \
  --model phi4_mini_instruct --methods alqr --devices cuda:0 \
  --alqr-lambda 1.5 --alqr-q 0.1 --alqr-r 1 --alqr-q-final 0.05
python -m robust_steerability.benchmarks.mgsm evaluate \
  --model phi4_mini_instruct --methods alqr --devices auto \
  --generation-batch-size 64 --replace-completed
python -m robust_steerability.benchmarks.mgsm score \
  --model phi4_mini_instruct --methods alqr --devices auto \
  --scorers default --api-concurrency 500 --api-batch-size 20 \
  --replace-completed
```

Use `--replace-completed` only on the first evaluation attempt. If an interrupted
evaluation is resumed, omit it so completed language shards are retained.
