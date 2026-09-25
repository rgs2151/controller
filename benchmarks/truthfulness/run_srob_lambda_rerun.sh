#!/usr/bin/env bash
set -euo pipefail

# Prepared launcher for the nine-model S-rob Truthfulness rerun.
# It intentionally excludes Spanish, MMLU, ALQR, fluency, and every API judge.
# Run explicitly with: bash benchmarks/truthfulness/run_srob_lambda_rerun.sh

calibration_id="lambda_txi_n200_r1"
devices="${TRUTHFULNESS_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
batch_size="${TRUTHFULNESS_BATCH_SIZE:-64}"
persist_cache_root="${TRUTHFULNESS_PERSIST_CACHE_ROOT:-}"
persist_repo_root="${TRUTHFULNESS_PERSIST_REPO_ROOT:-}"
default_models=(
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
models=("${default_models[@]}")
if (( $# > 0 )); then
  models=("$@")
fi

if [[ "${LIGHTNING_ARTIFACTS_DIR:-}" == /tmp/* ]]; then
  if [[ -z "$persist_cache_root" || -z "$persist_repo_root" ]]; then
    echo "Refusing an ephemeral run without Teamspace persistence roots." >&2
    exit 2
  fi
fi

copy_tree() {
  local source=$1
  local destination=$2
  mkdir -p "$destination"
  cp -a "$source/." "$destination/"
}

persist_model() {
  local model=$1
  local active_model_root="${LIGHTNING_ARTIFACTS_DIR}/robust-steering-cache/truthfulness/${model}"
  local active_calibration="${active_model_root}/calibrations/h_infinity/${calibration_id}"
  local active_evaluation="${active_model_root}/evaluations/kv_cache_off/calibrations/${calibration_id}"
  local active_results="benchmarks/truthfulness/results/kv_cache_off/calibrations/${calibration_id}/${model}"
  local durable_model_root="${persist_cache_root}/${model}"
  local durable_calibration="${durable_model_root}/calibrations/h_infinity/${calibration_id}"
  local durable_evaluation="${durable_model_root}/evaluations/kv_cache_off/calibrations/${calibration_id}"
  local durable_results="${persist_repo_root}/benchmarks/truthfulness/results/kv_cache_off/calibrations/${calibration_id}/${model}"

  local required=(
    "${active_calibration}/lambda_sweep/selection.json"
    "${active_calibration}/selection.json"
    "${active_evaluation}/results/truthfulness/original.json"
    "${active_evaluation}/results/truthfulness/h_infinity.json"
    "${active_results}/truthfulness/original.json"
    "${active_results}/truthfulness/h_infinity.json"
  )
  local path
  for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
      echo "Refusing to persist incomplete ${model} output; missing ${path}" >&2
      exit 3
    fi
  done

  copy_tree "$active_calibration" "$durable_calibration"
  copy_tree "$active_evaluation" "$durable_evaluation"
  copy_tree "$active_results" "$durable_results"

  mkdir -p "${persist_repo_root}/logs"
  shopt -s nullglob
  local log_directory
  for log_directory in logs/truthfulness-"${model}"-*; do
    cp -a "$log_directory" "${persist_repo_root}/logs/"
  done
  shopt -u nullglob

  local comparisons=(
    "${active_calibration}/lambda_sweep/selection.json:${durable_calibration}/lambda_sweep/selection.json"
    "${active_calibration}/selection.json:${durable_calibration}/selection.json"
    "${active_evaluation}/results/truthfulness/original.json:${durable_evaluation}/results/truthfulness/original.json"
    "${active_evaluation}/results/truthfulness/h_infinity.json:${durable_evaluation}/results/truthfulness/h_infinity.json"
    "${active_results}/truthfulness/original.json:${durable_results}/truthfulness/original.json"
    "${active_results}/truthfulness/h_infinity.json:${durable_results}/truthfulness/h_infinity.json"
  )
  local pair source destination
  for pair in "${comparisons[@]}"; do
    source=${pair%%:*}
    destination=${pair#*:}
    if ! cmp -s "$source" "$destination"; then
      echo "Persistent verification failed for ${destination}" >&2
      exit 4
    fi
  done

  python - "$model" "$durable_calibration/PERSISTED.json" <<'PY'
import datetime
import json
import pathlib
import subprocess
import sys

model, destination = sys.argv[1:]
payload = {
    "status": "complete_and_verified",
    "model": model,
    "calibration_id": "lambda_txi_n200_r1",
    "persisted_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
}
path = pathlib.Path(destination)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(path)
PY
  sync
  echo "Persisted and verified ${model} in Teamspace."
}

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

  persist_model "$model"
done
