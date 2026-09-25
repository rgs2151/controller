# Truthfulness S-rob lambda-first rerun

This rerun is isolated under calibration ID `lambda_txi_n200_r1`. It is prepared
but must not be launched until GPUs are explicitly assigned.

## Protocol

For each of the nine small models:

1. Reuse the existing Truthfulness artifacts, nominal dynamics, and Jacobians.
2. Sweep the H-infinity setpoint multiplier over
   `[0.1, 0.25, 0.5, 0.75, 1.0, 1.5]` at fixed
   `Q/R=0.1`, `Qf/R=0.1`, and `R=1.0`.
3. Select lambda using aggregate TruthfulQA `True x Informative` on the frozen
   200-question calibration set, with one decoding repetition and KV cache off.
4. At the selected lambda, run the existing 32-point `Q/R x Qf/R` sweep and
   select again using aggregate `True x Informative` only.
5. Evaluate only `Original` and `H-infinity` on English TruthfulQA, once.
6. Score only `truthfulqa_true` and `truthfulqa_informative`. No fluency,
   AXBench overall, Spanish, MMLU, or ALQR is run.

The historical run is preserved outside the repository at
`/home/dev/truthfulness_srob_pre_lambda_20260925`. The active artifacts remain in
place because the rerun depends on them.

## Four-GPU launch

From the repository root:

```bash
TRUTHFULNESS_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
TRUTHFULNESS_BATCH_SIZE=64 \
bash benchmarks/truthfulness/run_srob_lambda_rerun.sh
```

The launcher processes models sequentially while distributing each model's six
lambda candidates and 32 Q/Qf candidates across all four GPUs. Every stage is
resume-safe through the named calibration cache.
