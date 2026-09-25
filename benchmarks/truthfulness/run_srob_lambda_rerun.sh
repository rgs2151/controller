#!/usr/bin/env bash
set -euo pipefail

# Prepared launcher for the nine-model S-rob Truthfulness rerun.
# It intentionally excludes Spanish, MMLU, ALQR, fluency, and every API judge.
# Run explicitly with: bash benchmarks/truthfulness/run_srob_lambda_rerun.sh

calibration_id="lambda_txi_n200_r1"
devices="${TRUTHFULNESS_DEVICES:-0,1,2,3}"
batch_size="${TRUTHFULNESS_BATCH_SIZE:-64}"
models=(
  pythia_14m
  pythia_31m
  distilgpt2
  gpt2_small
  smollm2_135m
  pythia_160m
  gpt2_medium
  qwen25_05b
  gpt2_large
)

for model in "${models[@]}"; do
  python -m robust_steerability.benchmarks.truthfulness calibrate \
    --model "$model" \
    --run-profile srob_small_models \
    --methods h_infinity \
    --datasets id \
    --devices "$devices" \
    --calibration-id "$calibration_id" \
    --selection-metric truthfulqa_txi \
    --h-infinity-lambda-sweep \
    --generation-batch-size "$batch_size"

  python -m robust_steerability.benchmarks.truthfulness evaluate \
    --model "$model" \
    --run-profile srob_small_models \
    --methods original,h_infinity \
    --datasets id \
    --devices "$devices" \
    --calibration-id "$calibration_id" \
    --generation-batch-size "$batch_size" \
    --evaluation-repetitions 1

  python -m robust_steerability.benchmarks.truthfulness score \
    --model "$model" \
    --run-profile srob_small_models \
    --methods original,h_infinity \
    --datasets id \
    --scorers truthfulqa_true,truthfulqa_informative \
    --devices "$devices" \
    --calibration-id "$calibration_id" \
    --evaluation-repetitions 1
done

