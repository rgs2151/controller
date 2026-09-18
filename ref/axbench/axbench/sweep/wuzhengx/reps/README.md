# Preference Vector Suppression

This document provides commands for running inference and evaluation for preference vector suppression experiments with suppression on concept 500.
Data generation and training remains the same as positive steering. We directly apply trained vectors for suppression

## Data Generation
Run the following command to generate training data for suppression experiments. This is the same as positive steering:
```bash
python axbench/scripts/generate.py \
--config axbench/sweep/preference_vectors/experiments/p_vector_dps_g2-9b_axbench_suppress.yaml \
--dump_dir axbench/concept500
```

## Training
Run the following command to training for suppression experiments. This is the same as positive steering:

```bash
torchrun --nproc_per_node=$gpu_count --master_port=50000 axbench/scripts/train.py \
--config axbench/sweep/preference_vectors/experiments/p_vector_dps_g2-9b_axbench_suppress.yaml \
--dump_dir axbench/results_concept500/prod_9b_l31_suppress \
--overwrite_data_dir axbench/concept500/generate \
```

## Inference

Run the following command to perform inference for suppression experiments:

```bash
torchrun --nproc_per_node=$gpu_count --master_port=13121 /nlp/scr/qinanyu/preference-vectors/axbench/scripts/inference.py \
--config axbench/sweep/preference_vectors/experiments/p_vector_dps_g2-9b_axbench_suppress.yaml \
--mode steering \
--layer 31 \
--dump_dir  axbench/results_concept500/prod_9b_l31_suppress \
--overwrite_metadata_dir axbench/concept500/generate \
--overwrite_inference_dump_dir axbench/results_concept500/prod_9b_l31_suppress/inference_suppress \
--suppress_eval_dir axbench/concept500
```

## Evaluation

After running inference, evaluate the results with this command:

```bash
python axbench/scripts/evaluate.py \
--config axbench/sweep/preference_vectors/experiments/p_vector_dps_g2-9b_axbench_suppress.yaml \
--mode steering \
--dump_dir axbench/results_concept500/prod_9b_l31_suppress \
--overwrite_metadata_dir axbench/concept500/generate \
--overwrite_inference_dump_dir axbench/results_concept500/prod_9b_l31_suppress/inference_suppress \
--overwrite_evaluate_dump_dir axbench/results_concept500/prod_9b_l31_suppress/evaluate_suppress
```

## Parameters Explained

- `--mode steering`: Runs in steering mode for concept suppression
- `--layer 31`: Specifies which model layer to apply the intervention
- `--suppress_eval_dir`: Directory containing suppression evaluation data. The eval file will generate once and can be reloaded for furture evaluation


# Preference Vector Multishot Experiments

This document provides commands for running training, inference, and evaluation for preference vector multishot experiments with multishot attack.

```bash
python axbench/scripts/generate.py \
--config axbench/sweep/preference_vectors/experiments/p_vector_dps_g2-12b_axbench_suppress_rule.yaml \
--dump_dir axbench/concept_rule_20 \
--steer_data_type "rule"
```
## Training for rule base suppression 

Run the following command to train models for rule base suppression:

```bash
torchrun --master_port=12032 --nproc_per_node=$gpu_count axbench/scripts/train.py \
--config axbench/sweep/preference_vectors/experiments/p_vector_dps_g2-12b_axbench_suppress_rule.yaml \
--layer 22 \
--dump_dir axbench/concept_rule_20/prod_12b_l20_rule \
--overwrite_data_dir axbench/concept_rule_20/generate
```

## Inference on rule base suppression Data

First, run inference on suppression data to generate the factors for multishot experiments:

```bash
torchrun --nproc_per_node=$gpu_count --master_port=12041 axbench/scripts/inference.py \
--config axbench/sweep/preference_vectors/experiments/p_vector_dps_g2-12b_axbench_suppress_rule.yaml \
--mode steering \
--layer 22 \
--dump_dir axbench/concept_rule_20/prod_12b_l20_rule \
--overwrite_metadata_dir axbench/concept_rule_20/generate \
--overwrite_inference_dump_dir axbench/concept_rule_20/prod_12b_l20_rule/inference_suppression \
--suppress_eval_dir axbench/concept_rule_20
```

## Evaluation of rule base uppression

Evaluate the suppression results to generate the parquet file needed for multishot:

```bash
python axbench/scripts/evaluate.py \
--config axbench/sweep/preference_vectors/experiments/p_vector_dps_g2-12b_axbench_suppress_rule.yaml \
--mode steering \
--dump_dir axbench/concept_rule_20/prod_12b_l20_rule \
--overwrite_metadata_dir axbench/concept_rule_20/generate \
--overwrite_inference_dump_dir axbench/concept_rule_20/prod_12b_l20_rule/inference_suppression \
--overwrite_evaluate_dump_dir axbench/concept_rule_20/prod_12b_l20_rule/evaluate_suppression
```

## Inference on multishot rule base suppression

Run the following command to perform inference for multishot experiments using the factors from suppression evaluation:

```bash
torchrun --nproc_per_node=$gpu_count --master_port=12041 axbench/scripts/inference.py \
--config axbench/sweep/preference_vectors/experiments/p_vector_dps_g3-27b_axbench_attack.yaml \
--mode steering \
--layer 22 \
--dump_dir axbench/concept_rule_20/prod_12b_l20_rule \
--overwrite_metadata_dir axbench/concept_rule_20/generate \
--overwrite_inference_dump_dir axbench/concept_rule_20/prod_12b_l20_rule/inference_multishot \
--suppress_eval_dir axbench/concept_rule_20 \
--multishot_factors_parquet axbench/concept_rule_20/prod_12b_l20_rule/evaluate_suppression/steering_data.parquet
```

## Evaluation of multishot rule base suppression

After running inference, evaluate the multishot results with this command:

```bash
python axbench/scripts/evaluate.py \
--config axbench/sweep/preference_vectors/experiments/p_vector_dps_g2-9b_axbench_attack.yaml \
--mode steering \
--dump_dir axbench/concept_rule_20/prod_12b_l20_rule \
--overwrite_metadata_dir axbench/concept_rule_20/generate \
--overwrite_inference_dump_dir axbench/concept_rule_20/prod_12b_l20_rule/inference_multishot \
--overwrite_evaluate_dump_dir axbench/concept_rule_20/prod_12b_l20_rule/evaluate_multishot
```

## Parameters Explained

- `--mode steering`: Runs in steering mode for experiments
- `--layer`: Specifies which model layer to apply the intervention
- `--steering_batch_size`: Sets batch size for steering operations
- `--suppress_eval_dir`: Directory containing suppression evaluation data
- `--multishot_factors_parquet`: Path to the parquet file containing factors from previous suppression evaluation to select the best factors
- `--overwrite_inference_dump_dir`: Directory to save inference results
- `--overwrite_evaluate_dump_dir`: Directory to save evaluation results