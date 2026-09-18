# MGSM multilingual transfer benchmark

This benchmark constructs one English-to-Spanish residual direction from all 250
matched MGSM test-question pairs, then evaluates a fixed 100-problem subset in
five transfer languages: Chinese, French, Japanese, Swahili, and Telugu.

- `artifacts`: materialize the paired direction corpus, fit the per-layer DiffMean
  setpoint, and average 50 Spanish-question Jacobians into the nominal `A` shared
  by A-LQR and H∞.
- `calibrate`: write the frozen S-PID/A-LQR selections and select H∞ from a 12-point
  Q/R–Qf/R grid using 50 fixed GSM8K train questions and the AXBench three-score
  harmonic mean.
- `evaluate`: generate one native-eight-shot worked solution for each of 100
  matched problems in every requested input language, capped at 256 new tokens.
- `score`: independently compute exact final-number accuracy, AXBench Spanish rule
  following, instruction relevance, fluency, and their steering harmonic mean.

English and Spanish construct the direction and are not in the default evaluation.
The remaining registered languages stay independently runnable with `--datasets`;
evaluation never refits the controller by language.

```bash
python -m robust_steerability.benchmarks.mgsm artifacts --model qwen3_4b --devices auto
python -m robust_steerability.benchmarks.mgsm calibrate --model qwen3_4b --devices auto
python -m robust_steerability.benchmarks.mgsm evaluate --model qwen3_4b --devices auto
python -m robust_steerability.benchmarks.mgsm score --model qwen3_4b --scorers default --devices auto
```

Use `--datasets all` only for the deferred 11-language expansion.
