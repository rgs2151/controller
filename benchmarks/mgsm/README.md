# MGSM multilingual transfer benchmark

This benchmark constructs one English-to-Spanish residual direction from all 250
matched MGSM test-question pairs, then evaluates a fixed 100-problem subset in
five transfer languages: Chinese, French, Japanese, Swahili, and Telugu.

- `artifacts`: materialize the paired direction corpus, fit the per-layer DiffMean
  setpoint, and average 50 Spanish-question Jacobians into the nominal `A` shared
  by A-LQR and H∞.
- `calibrate`: write the frozen S-PID/A-LQR selections, optionally select H∞
  lambda first at fixed `Q/R=Qf/R=0.1` and `R=1`, then run the existing 12-point
  Q/R–Qf/R grid. Both stages reuse the same 50 frozen GSM8K questions and select
  by the AXBench concept-relevance, instruction-relevance, and fluency harmonic
  mean.
- `evaluate`: generate one native-eight-shot worked solution for each of 100
  matched problems in every requested input language, capped at 256 new tokens.
- `score`: independently compute exact final-number accuracy, AXBench Spanish rule
  following, instruction relevance, fluency, and their steering harmonic mean.

English and Spanish construct the direction and are not in the default evaluation.
The remaining registered languages stay independently runnable with `--datasets`;
evaluation never refits the controller by language.

Qwen3-4B is the completed first model. The next run uses
Llama-3.2-3B-Instruct with Original, S-PID, A-LQR, and H-infinity.

```bash
python -m robust_steerability.benchmarks.mgsm artifacts --model llama32_3b_instruct --methods original,spid,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm calibrate --model llama32_3b_instruct --methods original,spid,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm evaluate --model llama32_3b_instruct --methods original,spid,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm score --model llama32_3b_instruct --methods original,spid,alqr,h_infinity --scorers default --devices auto
```

Use `--datasets all` only for the deferred 11-language expansion.
