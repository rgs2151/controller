# L-CiteEval Small benchmark

This independent fast-iteration benchmark leaves `lciteeval` and
`lciteeval_spanish` unchanged. It tests whether an AXBench positive-sentiment
controller selected on 8K citation QA retains task quality at longer contexts.

- Model: Llama-3.2-1B-Instruct.
- Methods: Original, A-LQR, and H∞.
- Direction: AXBench concept 499, using the frozen 72 positive and 72 matched
  negative examples.
- Shared dynamics: 50 desired AXBench prompt Jacobians; A-LQR and H∞ consume
  the same saved nominal dynamics.
- H∞ disturbance fit: the frozen 200 AlpacaEval disturbance prompts.
- H∞ selection: 12 cost configurations on 10 frozen HotpotQA 8K prompts.
- Selection score: `0.40 × answer recall + 0.40 × citation F1 + 0.10 ×
  normalized fluency + 0.10 × normalized concept relevance`.
- Available evaluation conditions: the same 10 question identities at 8K, 16K,
  and 32K. The default run enables only 8K and 32K; 16K can be added later with
  `--datasets 16k` without changing the benchmark or calibration.
- Evaluation decoding: one deterministic generation per question, KV cache off,
  and a 128-token cap.
- Reporting: answer precision/recall/F1, citation precision/recall/F1, AXBench
  concept relevance, instruction relevance, fluency, and overall steering.

Run the four stages independently:

```bash
python -m robust_steerability.benchmarks.lciteeval_small artifacts --model llama32_1b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval_small calibrate --model llama32_1b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval_small evaluate --model llama32_1b_instruct --devices auto
python -m robust_steerability.benchmarks.lciteeval_small score --model llama32_1b_instruct --devices auto
```

The default dataset selection above is 8K and 32K. To append the currently
disabled 16K condition, run the evaluation and score stages with
`--datasets 16k`; 8K remains the sole H∞ selection condition.

Remote caches live under `~/robust-steering-cache/lciteeval_small/`; compact
results and logs remain Git-tracked under this unit and the repository `logs/`.
