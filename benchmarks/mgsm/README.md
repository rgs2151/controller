# MGSM multilingual transfer benchmark

This benchmark constructs one English-to-Spanish residual direction from all 250
matched MGSM test-question pairs, then evaluates a fixed 100-problem subset in
five transfer languages: Chinese, French, Japanese, Swahili, and Telugu.

- `artifacts`: materialize the paired direction corpus, fit the per-layer DiffMean
  setpoint, and average 50 Spanish-question Jacobians into the nominal `A` shared
  by A-LQR and H∞.
- `calibrate`: write the frozen S-PID/A-LQR selections and run the 12-point H∞
  Q/R–Qf/R grid at the same fixed `lambda=1.5` target used by A-LQR. The optional
  lambda-sweep machinery remains available but is disabled. H∞ fits its
  disturbance geometry on 200 translated GSM8K prompts balanced across Bengali,
  German, Russian, and Thai, then selects the cost grid on 50 additional disjoint
  prompts in the same four-language mixture. Selection uses the mean per-response
  balanced additive combination of exact-answer accuracy and normalized AXBench
  Overall.
- `evaluate`: generate one native-eight-shot worked solution for each of 100
  matched problems in every requested input language, capped at 256 new tokens.
- `score`: independently compute exact final-number accuracy, AXBench Spanish rule
  following, instruction relevance, fluency, and their steering harmonic mean.

English and Spanish construct the direction and are not in the default evaluation.
The remaining registered languages stay independently runnable with `--datasets`;
evaluation never refits the controller by language.

Qwen3-4B is complete under its original English calibration protocol. The first
Llama-3.2-3B-Instruct H∞ run exposed calibration saturation and is superseded by
the four-language calibration specified above. Original, S-PID, and A-LQR remain
unchanged; only the Llama H∞ rows are regenerated.

```bash
python -m robust_steerability.benchmarks.mgsm artifacts --model llama32_3b_instruct --methods original,spid,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm calibrate --model llama32_3b_instruct --methods original,spid,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm evaluate --model llama32_3b_instruct --methods original,spid,alqr,h_infinity --devices auto
python -m robust_steerability.benchmarks.mgsm score --model llama32_3b_instruct --methods original,spid,alqr,h_infinity --scorers default --devices auto
```

Use `--datasets all` only for the deferred 11-language expansion.
