# L-CiteEval Spanish-transfer benchmark

This is the current L-CiteEval pipeline. The historical positive-sentiment
Qwen run remains under `benchmarks/lciteeval/` and is never reused here.

- `artifacts`: fit the Spanish-minus-English MGSM direction and average 50
  Spanish-question Jacobians into the `A` shared by A-LQR and H∞.
- `calibrate`: fit H∞ disturbance geometry on 200 upstream 2WikiMultihopQA
  training prompts and select one of 12 cost configurations on the 40 official
  L-CiteEval 2Wiki base-context questions.
- `evaluate`: answer the same 40 HotpotQA questions at 8K and 16K with Original,
  S-PID, A-LQR, and H∞. The registered 32K condition is deferred.
- `score`: preserve raw Spanish generations, translate a scoring copy to English,
  compute answer/citation quality on that copy, and compute Spanish adherence,
  instruction relevance, and fluency on the raw response.

```bash
python -m robust_steerability.benchmarks.lciteeval artifacts --model llama31_8b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval calibrate --model llama31_8b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval evaluate --model llama31_8b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval score --model llama31_8b_instruct --scorers default --devices auto
```
