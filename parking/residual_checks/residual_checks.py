from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from robust_steerability.artifacts import configuration_hash
from robust_steerability.benchmarks.toxicity import (
    load_jigsaw_toxicity_prompts,
    load_real_toxicity_prompt_pools,
    toxicity_probabilities,
)
from robust_steerability.benchmarks.base import validate_disjoint_prompt_ids
from robust_steerability.calibration.residuals import (
    residual_metrics as package_residual_metrics,
)
from robust_steerability.calibration.targets import build_contrastive_target
from robust_steerability.control.lqr import LQRController, solve_identity_input_lqr
from robust_steerability.modeling.huggingface import (
    load_access_token,
    load_quantized_causal_model,
    load_sequence_classifier,
    tokenize_text,
)
from robust_steerability.modeling.interventions import (
    forward_with_policy,
    register_generation_policy_hooks,
)
from robust_steerability.modeling.jacobians import (
    capture_layer_inputs,
    layer_last_token_jacobian as package_layer_last_token_jacobian,
)
from robust_steerability.runtime.generation import write_generation_group
from robust_steerability.runtime.policy import SemanticSetpointPolicy
from robust_steerability.runtime.rollout import (
    read_rollout as package_read_rollout,
    write_prompt_rollout_group,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_DIR = Path(__file__).resolve().parent
CACHE_DIR = UNIT_DIR / "cache"
PLOTS_DIR = UNIT_DIR / "plots"
PROMPTS_PATH = CACHE_DIR / "prompts.json"
FIT_ACTIVATIONS_PATH = CACHE_DIR / "fit_activations.h5"
FEATURES_PATH = CACHE_DIR / "features.pt"
JACOBIAN_DIR = CACHE_DIR / "jacobians"
CONTROLLER_PATH = CACHE_DIR / "controller.pt"
ANALYSIS_CACHE_PATH = CACHE_DIR / "analysis.json"
TOXICITY_SCORES_PATH = CACHE_DIR / "toxicity_scores.json"
METRICS_PATH = PLOTS_DIR / "residual_checks_metrics.csv"
SUMMARY_PATH = PLOTS_DIR / "residual_checks_summary.json"
TOXICITY_METRICS_PATH = PLOTS_DIR / "generated_toxicity_metrics.csv"
TOXICITY_SUMMARY_PATH = PLOTS_DIR / "generated_toxicity_summary.json"

MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
RTP_ID = "allenai/real-toxicity-prompts"
RTP_REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
JIGSAW_ID = "tcapelle/jigsaw-toxic-comment-classification-challenge"
JIGSAW_REVISION = "2bf801de1b879f287943ecfc81fdca8690d9fc61"
TOXICITY_MODEL_ID = "s-nlp/roberta_toxicity_classifier"
TOXICITY_MODEL_REVISION = "048c25bb1e199b98802784f96325f4840f22145d"
SEED = 2151
N_FEATURE_PROMPTS = 200
N_JACOBIAN_PROMPTS = 50
N_EVAL_PROMPTS = 50
MAX_LENGTH = 512
JACOBIAN_MAX_LENGTH = 24
JACOBIAN_VJP_CHUNK = 256
Q = 0.1
R = 1.0
Q_FINAL = 1.0
LAMBDA = 2.5
EPSILON = 1e-12
BOOTSTRAP_SAMPLES = 2000
STATE_DEFINITION = "raw_decoder_layer_inputs_and_final_raw_output_v1"
GENERATION_STATE_DEFINITION = "teacher_forced_raw_states_with_generation_time_intervention_mask_v1"
MAX_NEW_TOKENS = 100
GENERATION_TOP_P = 0.3
GENERATION_TEMPERATURE = 1.0
GENERATION_REPETITION_PENALTY = 1.2

CONFIG = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "rtp_id": RTP_ID,
    "rtp_revision": RTP_REVISION,
    "jigsaw_id": JIGSAW_ID,
    "jigsaw_revision": JIGSAW_REVISION,
    "seed": SEED,
    "n_feature_prompts": N_FEATURE_PROMPTS,
    "n_jacobian_prompts": N_JACOBIAN_PROMPTS,
    "n_eval_prompts": N_EVAL_PROMPTS,
    "max_length": MAX_LENGTH,
    "jacobian_max_length": JACOBIAN_MAX_LENGTH,
    "q": Q,
    "r": R,
    "q_final": Q_FINAL,
    "lambda": LAMBDA,
}
CONFIG_HASH = configuration_hash(CONFIG)

GENERATION_CONFIG = {
    "base_config_hash": CONFIG_HASH,
    "toxicity_model_id": TOXICITY_MODEL_ID,
    "toxicity_model_revision": TOXICITY_MODEL_REVISION,
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": True,
    "top_p": GENERATION_TOP_P,
    "temperature": GENERATION_TEMPERATURE,
    "repetition_penalty": GENERATION_REPETITION_PENALTY,
    "seed": SEED,
    "state_definition": GENERATION_STATE_DEFINITION,
}
GENERATION_CONFIG_HASH = configuration_hash(GENERATION_CONFIG)

CONDITION_COLORS = {"id": "midnightblue", "ood": "darkred"}
CONDITION_LABELS = {"id": "RTP (ID)", "ood": "Jigsaw (OOD)"}


def load_hf_token() -> str:
    return load_access_token(REPO_ROOT)


def prepare_directories() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    JACOBIAN_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def stable_sample(records: list[dict[str, object]], count: int, rng: random.Random) -> list[dict[str, object]]:
    if len(records) < count:
        raise ValueError(f"Requested {count} records from a pool of {len(records)}")
    return [records[index] for index in rng.sample(range(len(records)), count)]


def prepare_prompts() -> dict[str, object]:
    prepare_directories()
    if PROMPTS_PATH.exists():
        payload = json.loads(PROMPTS_PATH.read_text())
        if payload["config_hash"] != CONFIG_HASH:
            raise ValueError(f"Incompatible prompt cache: {PROMPTS_PATH}")
        return payload

    all_rtp, toxic, nontoxic = load_real_toxicity_prompt_pools(
        RTP_ID,
        RTP_REVISION,
    )
    all_jigsaw = load_jigsaw_toxicity_prompts(JIGSAW_ID, JIGSAW_REVISION)

    rng = random.Random(SEED)
    fit_toxic = stable_sample(toxic, N_FEATURE_PROMPTS, rng)
    fit_nontoxic = stable_sample(nontoxic, N_FEATURE_PROMPTS, rng)
    jacobian_ids = {
        record["prompt_id"] for record in stable_sample(fit_nontoxic, N_JACOBIAN_PROMPTS, rng)
    }
    fit_ids = {record["prompt_id"] for record in fit_toxic + fit_nontoxic}
    id_pool = [record for record in all_rtp if record["prompt_id"] not in fit_ids]
    eval_id = stable_sample(id_pool, N_EVAL_PROMPTS, rng)
    eval_ood = stable_sample(all_jigsaw, N_EVAL_PROMPTS, rng)
    validate_disjoint_prompt_ids(
        fit_ids,
        set(),
        {record["prompt_id"] for record in eval_id + eval_ood},
    )

    for record in fit_nontoxic:
        record["jacobian_fit"] = record["prompt_id"] in jacobian_ids

    payload = {
        "config": CONFIG,
        "config_hash": CONFIG_HASH,
        "fit_toxic": fit_toxic,
        "fit_nontoxic": fit_nontoxic,
        "eval_id": eval_id,
        "eval_ood": eval_ood,
    }
    PROMPTS_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def load_model(device: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    return load_quantized_causal_model(
        MODEL_ID,
        MODEL_REVISION,
        device,
        load_hf_token(),
    )


def load_toxicity_model(
    device: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model, tokenizer = load_sequence_classifier(
        TOXICITY_MODEL_ID,
        TOXICITY_MODEL_REVISION,
        device,
        load_hf_token(),
    )
    if model.config.id2label != {0: "neutral", 1: "toxic"}:
        raise ValueError(f"Unexpected toxicity labels: {model.config.id2label}")
    return model, tokenizer


def tokenize(tokenizer: AutoTokenizer, text: str, device: torch.device, max_length: int) -> dict[str, torch.Tensor]:
    return tokenize_text(tokenizer, text, device, max_length)


def validate_h5(handle: h5py.File, purpose: str) -> None:
    if "config_hash" in handle.attrs and handle.attrs["config_hash"] != CONFIG_HASH:
        raise ValueError(f"Incompatible {purpose} cache")
    if len(handle) > 0 and handle.attrs.get("state_definition") != STATE_DEFINITION:
        raise ValueError(f"Incompatible activation-state definition in {purpose} cache")
    handle.attrs["config_hash"] = CONFIG_HASH
    handle.attrs["model_id"] = MODEL_ID
    handle.attrs["model_revision"] = MODEL_REVISION
    handle.attrs["state_definition"] = STATE_DEFINITION


def collect_fit_activations(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    prompts = prepare_prompts()
    records = [
        ("toxic", record) for record in prompts["fit_toxic"]
    ] + [
        ("nontoxic", record) for record in prompts["fit_nontoxic"]
    ]
    with h5py.File(FIT_ACTIVATIONS_PATH, "a") as handle:
        validate_h5(handle, "fit activation")
        for index, (label, record) in enumerate(records):
            group_name = f"{label}/{record['prompt_id'].replace(':', '_')}"
            if group_name in handle and bool(handle[group_name].attrs["complete"]):
                continue
            encoded = tokenize(tokenizer, str(record["text"]), model.device, MAX_LENGTH)
            output, states, controls = steered_forward(model, encoded, None)
            log_probabilities = output.logits[0, -1].float().log_softmax(dim=-1)
            next_token = int(torch.argmax(log_probabilities).item())
            parent = handle.require_group(label)
            short_name = str(record["prompt_id"]).replace(":", "_")
            write_prompt_rollout_group(
                parent,
                short_name,
                record,
                encoded["input_ids"],
                encoded["attention_mask"],
                states,
                controls,
                next_token,
                float(log_probabilities[next_token].item()),
            )
            print(f"fit activations {index + 1}/{len(records)}", flush=True)

    if FEATURES_PATH.exists():
        cached = torch.load(FEATURES_PATH, map_location="cpu", weights_only=False)
        if cached["config_hash"] != CONFIG_HASH or cached["state_definition"] != STATE_DEFINITION:
            raise ValueError(f"Incompatible feature cache: {FEATURES_PATH}")
        return

    means: dict[str, torch.Tensor] = {}
    with h5py.File(FIT_ACTIVATIONS_PATH, "r") as handle:
        for label in ["toxic", "nontoxic"]:
            names = sorted(handle[label].keys())
            total = None
            for name in names:
                last_token_states = torch.from_numpy(handle[label][name]["states"][:, -1, :]).double()
                total = last_token_states if total is None else total + last_token_states
            means[label] = (total / len(names)).float()

    target = build_contrastive_target(
        means["nontoxic"],
        means["toxic"],
        LAMBDA,
        EPSILON,
    )
    torch.save(
        {
            "config": CONFIG,
            "config_hash": CONFIG_HASH,
            "state_definition": STATE_DEFINITION,
            "nominal": target["nominal"],
            "toxic_mean": target["opposite_mean"],
            "feature": target["feature"],
            "feature_norm": target["feature_norm"],
            "feature_unit": target["feature_unit"],
            "beta": target["beta"],
        },
        FEATURES_PATH,
    )


def layer_last_token_jacobian(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    layer_kwargs: dict[str, object],
) -> torch.Tensor:
    return package_layer_last_token_jacobian(
        layer,
        hidden_states,
        layer_kwargs,
        JACOBIAN_VJP_CHUNK,
    )


def collect_jacobians(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    shard_index: int,
    shard_count: int,
) -> None:
    prompts = prepare_prompts()
    records = [record for record in prompts["fit_nontoxic"] if record["jacobian_fit"]]
    if len(records) != N_JACOBIAN_PROMPTS:
        raise ValueError("Prompt cache does not contain the expected Jacobian-fit records")
    for global_index, record in enumerate(records):
        if global_index % shard_count != shard_index:
            continue
        output_path = JACOBIAN_DIR / f"{global_index:04d}.pt"
        if output_path.exists():
            cached = torch.load(output_path, map_location="cpu", weights_only=False)
            if cached["config_hash"] != CONFIG_HASH or cached["prompt_id"] != record["prompt_id"]:
                raise ValueError(f"Incompatible Jacobian cache: {output_path}")
            continue
        encoded = tokenize(tokenizer, str(record["text"]), model.device, JACOBIAN_MAX_LENGTH)
        captured = capture_layer_inputs(model, encoded)
        prompt_jacobians = []
        for layer_index, (hidden_states, layer_kwargs) in enumerate(captured):
            jacobian = layer_last_token_jacobian(
                model.model.layers[layer_index],
                hidden_states,
                layer_kwargs,
            )
            prompt_jacobians.append(jacobian)
            print(
                f"jacobian prompt {global_index + 1}/{len(records)} layer {layer_index + 1}/{len(captured)}",
                flush=True,
            )
        torch.save(
            {
                "config_hash": CONFIG_HASH,
                "prompt_id": record["prompt_id"],
                "jacobians": torch.stack(prompt_jacobians).to(torch.float16),
            },
            output_path,
        )


def solve_lqr(jacobians: torch.Tensor, device: str) -> torch.Tensor:
    return solve_identity_input_lqr(
        jacobians,
        device,
        state_cost=Q,
        control_cost=R,
        terminal_cost=Q_FINAL,
    )


def build_controller(device: str) -> None:
    if CONTROLLER_PATH.exists():
        cached = torch.load(CONTROLLER_PATH, map_location="cpu", weights_only=False)
        if cached["config_hash"] != CONFIG_HASH or cached["state_definition"] != STATE_DEFINITION:
            raise ValueError(f"Incompatible controller cache: {CONTROLLER_PATH}")
        return
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature cache: {FEATURES_PATH}")
    paths = [JACOBIAN_DIR / f"{index:04d}.pt" for index in range(N_JACOBIAN_PROMPTS)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Jacobian caches; first missing: {missing[0]}")
    jacobian_sum = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["config_hash"] != CONFIG_HASH:
            raise ValueError(f"Incompatible Jacobian cache: {path}")
        jacobians = payload["jacobians"].float()
        jacobian_sum = jacobians if jacobian_sum is None else jacobian_sum + jacobians
    mean_jacobian = jacobian_sum / len(paths)
    gains = solve_lqr(mean_jacobian, device)
    features = torch.load(FEATURES_PATH, map_location="cpu", weights_only=False)
    torch.save(
        {
            **features,
            "jacobians": mean_jacobian,
            "gains": gains,
        },
        CONTROLLER_PATH,
    )


def steered_forward(
    model: AutoModelForCausalLM,
    encoded: dict[str, torch.Tensor],
    controller: dict[str, object] | None,
    steer_start_index: int | None = None,
) -> tuple[object, torch.Tensor, torch.Tensor]:
    policy = (
        SemanticSetpointPolicy(
            controller=LQRController.from_tracking_gains(controller["gains"]),
            feature_unit=controller["feature_unit"],
            setpoints=controller["beta"],
        )
        if controller is not None
        else None
    )
    return forward_with_policy(model, encoded, policy, steer_start_index)


def collect_rollout_file(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict[str, object]],
    condition: str,
    controller_name: str,
    controller: dict[str, object] | None,
) -> None:
    path = CACHE_DIR / f"rollouts_{condition}_{controller_name}.h5"
    with h5py.File(path, "a") as handle:
        validate_h5(handle, f"{condition} {controller_name} rollout")
        handle.attrs["condition"] = condition
        handle.attrs["controller"] = controller_name
        for index, record in enumerate(records):
            group_name = f"{index:04d}"
            if group_name in handle and bool(handle[group_name].attrs["complete"]):
                continue
            encoded = tokenize(tokenizer, str(record["text"]), model.device, MAX_LENGTH)
            output, states, controls = steered_forward(model, encoded, controller)
            log_probabilities = output.logits[0, -1].float().log_softmax(dim=-1)
            next_token = int(torch.argmax(log_probabilities).item())
            write_prompt_rollout_group(
                handle,
                group_name,
                record,
                encoded["input_ids"],
                encoded["attention_mask"],
                states,
                controls,
                next_token,
                float(log_probabilities[next_token].item()),
            )
            print(
                f"{condition} {controller_name} rollout {index + 1}/{len(records)}",
                flush=True,
            )


def collect_rollouts(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    if not CONTROLLER_PATH.exists():
        raise FileNotFoundError(f"Missing controller cache: {CONTROLLER_PATH}")
    prompts = prepare_prompts()
    controller = torch.load(CONTROLLER_PATH, map_location="cpu", weights_only=False)
    if controller["config_hash"] != CONFIG_HASH or controller["state_definition"] != STATE_DEFINITION:
        raise ValueError(f"Incompatible controller cache: {CONTROLLER_PATH}")
    for condition, records in [("id", prompts["eval_id"]), ("ood", prompts["eval_ood"])]:
        collect_rollout_file(model, tokenizer, records, condition, "baseline", None)
        collect_rollout_file(model, tokenizer, records, condition, "alqr", controller)


def register_generation_steering_hooks(
    model: AutoModelForCausalLM,
    controller: dict[str, object],
) -> list[torch.utils.hooks.RemovableHandle]:
    policy = SemanticSetpointPolicy(
        controller=LQRController.from_tracking_gains(controller["gains"]),
        feature_unit=controller["feature_unit"],
        setpoints=controller["beta"],
    )
    return register_generation_policy_hooks(model, policy)


def validate_generation_h5(handle: h5py.File, condition: str, controller_name: str) -> None:
    if "generation_config_hash" in handle.attrs:
        if handle.attrs["generation_config_hash"] != GENERATION_CONFIG_HASH:
            raise ValueError(f"Incompatible generation cache for {condition} {controller_name}")
    if len(handle) > 0 and handle.attrs.get("state_definition") != GENERATION_STATE_DEFINITION:
        raise ValueError(f"Incompatible generation-state definition for {condition} {controller_name}")
    handle.attrs["generation_config_hash"] = GENERATION_CONFIG_HASH
    handle.attrs["condition"] = condition
    handle.attrs["controller"] = controller_name
    handle.attrs["model_id"] = MODEL_ID
    handle.attrs["model_revision"] = MODEL_REVISION
    handle.attrs["state_definition"] = GENERATION_STATE_DEFINITION


def collect_generation_file(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict[str, object]],
    condition: str,
    controller_name: str,
    controller: dict[str, object] | None,
) -> None:
    path = CACHE_DIR / f"generations_{condition}_{controller_name}.h5"
    with h5py.File(path, "a") as handle:
        validate_generation_h5(handle, condition, controller_name)
        for index, record in enumerate(records):
            group_name = f"{index:04d}"
            if group_name in handle and bool(handle[group_name].attrs["complete"]):
                continue
            encoded = tokenize(tokenizer, str(record["text"]), model.device, MAX_LENGTH)
            prompt_length = int(encoded["input_ids"].shape[1])
            sample_seed = SEED + index + (100_000 if condition == "ood" else 0)
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            handles = register_generation_steering_hooks(model, controller) if controller is not None else []
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    return_dict_in_generate=True,
                    do_sample=True,
                    top_p=GENERATION_TOP_P,
                    repetition_penalty=GENERATION_REPETITION_PENALTY,
                    temperature=GENERATION_TEMPERATURE,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for hook_handle in handles:
                hook_handle.remove()
            sequences = generated.sequences
            generated_ids = sequences[:, prompt_length:]
            completion = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            full_encoded = {
                "input_ids": sequences.to(model.device),
                "attention_mask": torch.ones_like(sequences, device=model.device),
            }
            _, states, controls = steered_forward(
                model,
                full_encoded,
                controller,
                steer_start_index=prompt_length - 1,
            )
            write_generation_group(
                handle,
                group_name,
                record,
                sample_seed,
                prompt_length,
                sequences,
                generated_ids,
                states,
                controls,
                completion,
            )
            print(
                f"{condition} {controller_name} generation {index + 1}/{len(records)}",
                flush=True,
            )


def collect_generations(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    if not CONTROLLER_PATH.exists():
        raise FileNotFoundError(f"Missing controller cache: {CONTROLLER_PATH}")
    prompts = prepare_prompts()
    controller = torch.load(CONTROLLER_PATH, map_location="cpu", weights_only=False)
    if controller["config_hash"] != CONFIG_HASH or controller["state_definition"] != STATE_DEFINITION:
        raise ValueError(f"Incompatible controller cache: {CONTROLLER_PATH}")
    for condition, records in [("id", prompts["eval_id"]), ("ood", prompts["eval_ood"])]:
        collect_generation_file(model, tokenizer, records, condition, "baseline", None)
        collect_generation_file(model, tokenizer, records, condition, "alqr", controller)


def read_generation_texts() -> list[dict[str, object]]:
    rows = []
    for condition in ["id", "ood"]:
        for controller_name in ["baseline", "alqr"]:
            path = CACHE_DIR / f"generations_{condition}_{controller_name}.h5"
            with h5py.File(path, "r") as handle:
                if handle.attrs["generation_config_hash"] != GENERATION_CONFIG_HASH:
                    raise ValueError(f"Incompatible generation cache: {path}")
                complete_names = [name for name in sorted(handle) if not name.startswith("tmp_")]
                if len(complete_names) != N_EVAL_PROMPTS:
                    raise ValueError(
                        f"Expected {N_EVAL_PROMPTS} complete records in {path}, found {len(complete_names)}"
                    )
                for name in complete_names:
                    group = handle[name]
                    rows.append(
                        {
                            "prompt_id": str(group.attrs["prompt_id"]),
                            "condition": condition,
                            "controller": controller_name,
                            "completion": str(group.attrs["completion"]),
                            "generated_tokens": int(group["generated_ids"].shape[1]),
                        }
                    )
    return rows


def score_generations(device: str) -> list[dict[str, object]]:
    if TOXICITY_SCORES_PATH.exists():
        payload = json.loads(TOXICITY_SCORES_PATH.read_text())
        if payload["generation_config_hash"] != GENERATION_CONFIG_HASH:
            raise ValueError(f"Incompatible toxicity-score cache: {TOXICITY_SCORES_PATH}")
        return payload["rows"]
    rows = read_generation_texts()
    model, tokenizer = load_toxicity_model(device)
    probabilities = toxicity_probabilities(
        [str(row["completion"]) for row in rows],
        model,
        tokenizer,
        device,
        batch_size=16,
        max_length=512,
    )
    for row, probability in zip(rows, probabilities, strict=True):
        row["toxicity_probability"] = float(probability)
    payload = {
        "generation_config": GENERATION_CONFIG,
        "generation_config_hash": GENERATION_CONFIG_HASH,
        "rows": rows,
    }
    TOXICITY_SCORES_PATH.write_text(json.dumps(payload, indent=2))
    return rows


def read_rollout(path: Path) -> list[dict[str, object]]:
    return package_read_rollout(path, CONFIG_HASH)


def residual_metrics(
    states: torch.Tensor,
    controls: torch.Tensor,
    controller: dict[str, object],
) -> dict[str, object]:
    return package_residual_metrics(states, controls, controller, EPSILON)


def bootstrap_correlation_difference(
    frame: pd.DataFrame,
    first_measure: str,
    second_measure: str,
    outcome: str,
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, len(frame), len(frame))
        sample = frame.iloc[indices]
        first_rho = spearmanr(sample[first_measure], sample[outcome]).statistic
        second_rho = spearmanr(sample[second_measure], sample[outcome]).statistic
        values.append(float(first_rho - second_rho))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def compute_analysis() -> tuple[pd.DataFrame, dict[str, object], dict[str, np.ndarray]]:
    controller = torch.load(CONTROLLER_PATH, map_location="cpu", weights_only=False)
    rows = []
    layer_profiles: dict[str, np.ndarray] = {}
    for condition in ["id", "ood"]:
        for controller_name in ["baseline", "alqr"]:
            path = CACHE_DIR / f"rollouts_{condition}_{controller_name}.h5"
            records = read_rollout(path)
            if len(records) != N_EVAL_PROMPTS:
                raise ValueError(f"Expected {N_EVAL_PROMPTS} complete records in {path}, found {len(records)}")
            profiles = []
            for record in records:
                metrics = residual_metrics(record["states"], record["controls"], controller)
                profiles.append(metrics.pop("layer_relative_residual"))
                if controller_name == "alqr":
                    rows.append(
                        {
                            "prompt_id": record["prompt_id"],
                            "condition": condition,
                            **metrics,
                        }
                    )
            layer_profiles[f"{condition}_{controller_name}"] = np.mean(np.stack(profiles), axis=0)

    frame = pd.DataFrame(rows)
    rho_magnitude = spearmanr(frame["residual_magnitude"], frame["failure"])
    rho_directional = spearmanr(frame["directional_effect"], frame["failure"])
    id_amplification = frame.loc[frame["condition"] == "id", "amplification"]
    ood_amplification = frame.loc[frame["condition"] == "ood", "amplification"]
    amplification_test = mannwhitneyu(ood_amplification, id_amplification, alternative="greater")
    correlation_difference_interval = bootstrap_correlation_difference(
        frame,
        "directional_effect",
        "residual_magnitude",
        "failure",
    )
    condition_correlations = {}
    for condition in ["id", "ood"]:
        subset = frame[frame["condition"] == condition]
        condition_correlations[condition] = {}
        for measure in ["residual_magnitude", "directional_effect", "amplification"]:
            correlation = spearmanr(subset[measure], subset["failure"])
            condition_correlations[condition][measure] = {
                "rho": float(correlation.statistic),
                "p_value": float(correlation.pvalue),
            }
    summary = {
        "config": CONFIG,
        "n_id": int((frame["condition"] == "id").sum()),
        "n_ood": int((frame["condition"] == "ood").sum()),
        "spearman_residual_magnitude_vs_failure": {
            "rho": float(rho_magnitude.statistic),
            "p_value": float(rho_magnitude.pvalue),
        },
        "spearman_directional_effect_vs_failure": {
            "rho": float(rho_directional.statistic),
            "p_value": float(rho_directional.pvalue),
        },
        "spearman_rho_difference_directional_minus_magnitude": float(
            rho_directional.statistic - rho_magnitude.statistic
        ),
        "spearman_rho_difference_bootstrap_95_ci": list(correlation_difference_interval),
        "amplification_median": {
            "id": float(id_amplification.median()),
            "ood": float(ood_amplification.median()),
        },
        "amplification_ood_to_id_median_ratio": float(
            ood_amplification.median() / id_amplification.median()
        ),
        "mann_whitney_ood_greater_than_id": {
            "u": float(amplification_test.statistic),
            "p_value": float(amplification_test.pvalue),
        },
        "spearman_vs_failure_by_condition": condition_correlations,
        "empirical_gamma_95": {
            "id": float(id_amplification.quantile(0.95)),
            "ood": float(ood_amplification.quantile(0.95)),
        },
    }
    return frame, summary, layer_profiles


def compute_toxicity_analysis(
    residual_frame: pd.DataFrame,
    classifier_device: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    score_frame = pd.DataFrame(score_generations(classifier_device))
    toxicity = score_frame.pivot(
        index=["prompt_id", "condition"],
        columns="controller",
        values="toxicity_probability",
    ).reset_index()
    toxicity.columns.name = None
    toxicity = toxicity.rename(
        columns={"baseline": "baseline_toxicity", "alqr": "alqr_toxicity"}
    )
    token_counts = score_frame.pivot(
        index=["prompt_id", "condition"],
        columns="controller",
        values="generated_tokens",
    ).reset_index()
    token_counts.columns.name = None
    token_counts = token_counts.rename(
        columns={
            "baseline": "baseline_generated_tokens",
            "alqr": "alqr_generated_tokens",
        }
    )
    frame = toxicity.merge(token_counts, on=["prompt_id", "condition"], validate="one_to_one")
    frame = frame.merge(residual_frame, on=["prompt_id", "condition"], validate="one_to_one")
    frame["toxicity_shortfall"] = frame["alqr_toxicity"] - frame["baseline_toxicity"]
    frame["toxicity_reduction"] = -frame["toxicity_shortfall"]

    magnitude_rho = spearmanr(frame["residual_magnitude"], frame["toxicity_shortfall"])
    directional_rho = spearmanr(frame["directional_effect"], frame["toxicity_shortfall"])
    amplification_rho = spearmanr(frame["amplification"], frame["toxicity_shortfall"])
    correlation_difference_interval = bootstrap_correlation_difference(
        frame,
        "directional_effect",
        "residual_magnitude",
        "toxicity_shortfall",
    )
    condition_summaries = {}
    for condition in ["id", "ood"]:
        subset = frame[frame["condition"] == condition]
        paired_test = wilcoxon(
            subset["baseline_toxicity"],
            subset["alqr_toxicity"],
            alternative="greater",
        )
        condition_summaries[condition] = {
            "baseline_mean": float(subset["baseline_toxicity"].mean()),
            "alqr_mean": float(subset["alqr_toxicity"].mean()),
            "median_toxicity_reduction": float(subset["toxicity_reduction"].median()),
            "baseline_toxic_fraction_at_0.5": float((subset["baseline_toxicity"] >= 0.5).mean()),
            "alqr_toxic_fraction_at_0.5": float((subset["alqr_toxicity"] >= 0.5).mean()),
            "wilcoxon_baseline_greater_than_alqr": {
                "statistic": float(paired_test.statistic),
                "p_value": float(paired_test.pvalue),
            },
        }
    id_shortfall = frame.loc[frame["condition"] == "id", "toxicity_shortfall"]
    ood_shortfall = frame.loc[frame["condition"] == "ood", "toxicity_shortfall"]
    shift_test = mannwhitneyu(ood_shortfall, id_shortfall, alternative="greater")
    summary = {
        "generation_config": GENERATION_CONFIG,
        "n_id": int((frame["condition"] == "id").sum()),
        "n_ood": int((frame["condition"] == "ood").sum()),
        "condition_summary": condition_summaries,
        "spearman_residual_magnitude_vs_toxicity_shortfall": {
            "rho": float(magnitude_rho.statistic),
            "p_value": float(magnitude_rho.pvalue),
        },
        "spearman_directional_effect_vs_toxicity_shortfall": {
            "rho": float(directional_rho.statistic),
            "p_value": float(directional_rho.pvalue),
        },
        "spearman_amplification_vs_toxicity_shortfall": {
            "rho": float(amplification_rho.statistic),
            "p_value": float(amplification_rho.pvalue),
        },
        "spearman_rho_difference_directional_minus_magnitude": float(
            directional_rho.statistic - magnitude_rho.statistic
        ),
        "spearman_rho_difference_bootstrap_95_ci": list(correlation_difference_interval),
        "mann_whitney_ood_shortfall_greater_than_id": {
            "u": float(shift_test.statistic),
            "p_value": float(shift_test.pvalue),
        },
    }
    return frame, summary


def setup_style() -> None:
    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["patch.linewidth"] = 0
    plt.rcParams["image.interpolation"] = "none"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.format"] = "pdf"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False


def endpoint_ticks(axis, values: np.ndarray, dimension: str) -> None:
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    if dimension == "x":
        axis.set_xlim(lower, upper)
        axis.set_xticks([lower, upper])
    else:
        axis.set_ylim(lower, upper)
        axis.set_yticks([lower, upper])


def save_single_plot(fig: plt.Figure, axis: plt.Axes, stem: str) -> None:
    axis.title.set_fontsize(14)
    axis.xaxis.label.set_size(11)
    axis.yaxis.label.set_size(11)
    axis.tick_params(axis="both", labelsize=9)
    axis.set_box_aspect(1)
    sns.despine(ax=axis, trim=True, offset=10)
    fig.savefig(PLOTS_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(
        PLOTS_DIR / f"{stem}.png",
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
        dpi=220,
    )
    plt.close(fig)


def plot_analysis(frame: pd.DataFrame, summary: dict[str, object], profiles: dict[str, np.ndarray]) -> None:
    setup_style()

    fig, axis = plt.subplots(figsize=(3.5, 3.2))
    depth = np.linspace(0, 1, len(profiles["id_alqr"]))
    for condition in ["id", "ood"]:
        axis.plot(
            depth,
            profiles[f"{condition}_baseline"],
            color=CONDITION_COLORS[condition],
            linestyle=":",
            label=f"{CONDITION_LABELS[condition]}, baseline",
        )
        axis.plot(
            depth,
            profiles[f"{condition}_alqr"],
            color=CONDITION_COLORS[condition],
            linestyle="-",
            label=f"{CONDITION_LABELS[condition]}, A-LQR",
        )
    axis.set_title("Layer residuals")
    axis.set_xlabel("Normalized depth")
    axis.set_ylabel(r"Mean $\|\xi_k\|_2/\|x_{k+1}\|_2$")
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 1])
    profile_values = np.concatenate(list(profiles.values()))
    axis.set_ylim(0, float(profile_values.max()) * 1.05)
    axis.set_yticks([0, float(profile_values.max())])
    axis.legend(loc="best", fontsize=7)
    save_single_plot(fig, axis, "layer_residuals")

    fig, axis = plt.subplots(figsize=(3.2, 3.2))
    sns.stripplot(
        data=frame,
        x="condition",
        y="amplification",
        hue="condition",
        order=["id", "ood"],
        palette=CONDITION_COLORS,
        legend=False,
        jitter=0.18,
        size=4,
        alpha=0.7,
        ax=axis,
    )
    medians = frame.groupby("condition")["amplification"].median()
    for position, condition in enumerate(["id", "ood"]):
        axis.plot([position - 0.22, position + 0.22], [medians[condition]] * 2, color="black")
    ratio = summary["amplification_ood_to_id_median_ratio"]
    shift_p = summary["mann_whitney_ood_greater_than_id"]["p_value"]
    axis.set_title("Distribution shift amplifies residuals")
    axis.set_xlabel("")
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["RTP\n(ID)", "Jigsaw\n(OOD)"])
    axis.set_ylabel("Closed-loop residual amplification")
    axis.set_ylim(0, float(frame["amplification"].max()) * 1.05)
    axis.set_yticks([0, float(frame["amplification"].max())])
    axis.text(
        0.04,
        0.97,
        rf"OOD median = {ratio:.2f}$\times$ ID" + "\n" + rf"$p={shift_p:.1e}$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    save_single_plot(fig, axis, "residual_amplification")

    ood = frame[frame["condition"] == "ood"]
    ood_correlations = summary["spearman_vs_failure_by_condition"]["ood"]
    failure_max = float(ood["failure"].max())

    fig, axis = plt.subplots(figsize=(3.2, 3.2))
    axis.scatter(
        np.log10(ood["residual_magnitude"] + EPSILON),
        ood["failure"],
        color=CONDITION_COLORS["ood"],
        s=18,
        alpha=0.75,
    )
    magnitude_rho = ood_correlations["residual_magnitude"]["rho"]
    magnitude_p = ood_correlations["residual_magnitude"]["p_value"]
    axis.set_title(rf"OOD: residual size, $\rho={magnitude_rho:.2f}$")
    axis.set_xlabel(r"$\log_{10}\|\xi\|_2$")
    axis.set_ylabel("Semantic tracking failure")
    endpoint_ticks(axis, np.log10(ood["residual_magnitude"].to_numpy() + EPSILON), "x")
    axis.set_ylim(0, failure_max * 1.05)
    axis.set_yticks([0, failure_max])
    axis.text(0.04, 0.96, rf"$p={magnitude_p:.2g}$", transform=axis.transAxes, ha="left", va="top", fontsize=8)
    save_single_plot(fig, axis, "residual_magnitude_vs_failure")

    fig, axis = plt.subplots(figsize=(3.2, 3.2))
    axis.scatter(
        np.log10(ood["directional_effect"] + EPSILON),
        ood["failure"],
        color=CONDITION_COLORS["ood"],
        s=18,
        alpha=0.75,
    )
    directional_rho = ood_correlations["directional_effect"]["rho"]
    directional_p = ood_correlations["directional_effect"]["p_value"]
    axis.set_title(rf"OOD: direction-aware effect, $\rho={directional_rho:.2f}$")
    axis.set_xlabel(r"$\log_{10}|T_{\rm LQR}\xi|$")
    axis.set_ylabel("Semantic tracking failure")
    endpoint_ticks(axis, np.log10(ood["directional_effect"].to_numpy() + EPSILON), "x")
    axis.set_ylim(0, failure_max * 1.05)
    axis.set_yticks([0, failure_max])
    axis.text(0.04, 0.96, rf"$p={directional_p:.1e}$", transform=axis.transAxes, ha="left", va="top", fontsize=8)
    save_single_plot(fig, axis, "directional_effect_vs_failure")

    fig, axis = plt.subplots(figsize=(3.2, 3.2))
    axis.scatter(
        np.log10(ood["amplification"] + EPSILON),
        ood["failure"],
        color=CONDITION_COLORS["ood"],
        s=18,
        alpha=0.75,
    )
    amplification_rho = ood_correlations["amplification"]["rho"]
    amplification_p = ood_correlations["amplification"]["p_value"]
    axis.set_title(rf"OOD: amplification predicts failure, $\rho={amplification_rho:.2f}$")
    axis.set_xlabel("$\log_{10}$ residual amplification")
    axis.set_ylabel("Semantic tracking failure")
    endpoint_ticks(axis, np.log10(ood["amplification"].to_numpy() + EPSILON), "x")
    axis.set_ylim(0, failure_max * 1.05)
    axis.set_yticks([0, failure_max])
    axis.text(0.04, 0.96, rf"$p={amplification_p:.1e}$", transform=axis.transAxes, ha="left", va="top", fontsize=8)
    save_single_plot(fig, axis, "amplification_vs_failure")


def plot_toxicity(frame: pd.DataFrame, summary: dict[str, object]) -> None:
    setup_style()

    fig, axis = plt.subplots(figsize=(4.8, 3.2))
    positions = {("id", "baseline"): 0.0, ("id", "alqr"): 1.0, ("ood", "baseline"): 3.0, ("ood", "alqr"): 4.0}
    for _, row in frame.iterrows():
        condition = str(row["condition"])
        color = CONDITION_COLORS[condition]
        x_values = [positions[(condition, "baseline")], positions[(condition, "alqr")]]
        y_values = [row["baseline_toxicity"], row["alqr_toxicity"]]
        axis.plot(x_values, y_values, color=color, alpha=0.16, linewidth=0.6)
        axis.scatter(x_values[0], y_values[0], facecolors="white", edgecolors=color, s=14, linewidths=0.7)
        axis.scatter(x_values[1], y_values[1], color=color, s=14, alpha=0.75)
    for condition in ["id", "ood"]:
        subset = frame[frame["condition"] == condition]
        for controller_name, column in [("baseline", "baseline_toxicity"), ("alqr", "alqr_toxicity")]:
            position = positions[(condition, controller_name)]
            median = float(subset[column].median())
            axis.plot([position - 0.22, position + 0.22], [median, median], color="black")
    axis.set_title("Generated-text toxicity")
    axis.set_xlabel("")
    axis.set_ylabel("Toxic probability")
    axis.set_xticks([0, 1, 3, 4])
    axis.set_xticklabels(["RTP\nbaseline", "RTP\nA-LQR", "Jigsaw\nbaseline", "Jigsaw\nA-LQR"])
    axis.set_ylim(0, 1)
    axis.set_yticks([0, 1])
    save_single_plot(fig, axis, "generated_toxicity")

    shortfall_limits = frame["toxicity_shortfall"].to_numpy()
    for measure, label, stem, summary_key in [
        (
            "residual_magnitude",
            r"$\log_{10}\|\xi\|_2$",
            "residual_magnitude_vs_toxicity_shortfall",
            "spearman_residual_magnitude_vs_toxicity_shortfall",
        ),
        (
            "directional_effect",
            r"$\log_{10}|T_{\rm LQR}\xi|$",
            "directional_effect_vs_toxicity_shortfall",
            "spearman_directional_effect_vs_toxicity_shortfall",
        ),
    ]:
        fig, axis = plt.subplots(figsize=(3.2, 3.2))
        for condition in ["id", "ood"]:
            subset = frame[frame["condition"] == condition]
            axis.scatter(
                np.log10(subset[measure] + EPSILON),
                subset["toxicity_shortfall"],
                color=CONDITION_COLORS[condition],
                s=16,
                alpha=0.7,
                label=CONDITION_LABELS[condition],
            )
        rho = summary[summary_key]["rho"]
        title = "Residual magnitude" if measure == "residual_magnitude" else "Direction-aware effect"
        axis.set_title(rf"{title}, $\rho={rho:.2f}$")
        axis.set_xlabel(label)
        axis.set_ylabel("Toxicity change\n(A-LQR - baseline)")
        axis.axhline(0, color="black", linestyle=":", linewidth=0.7)
        axis.legend(loc="best", fontsize=7)
        endpoint_ticks(axis, np.log10(frame[measure].to_numpy() + EPSILON), "x")
        endpoint_ticks(axis, shortfall_limits, "y")
        save_single_plot(fig, axis, stem)


def analyze() -> None:
    frame, summary, profiles = compute_analysis()
    frame.to_csv(METRICS_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    ANALYSIS_CACHE_PATH.write_text(
        json.dumps(
            {
                "config_hash": CONFIG_HASH,
                "summary": summary,
                "layer_profiles": {name: values.tolist() for name, values in profiles.items()},
            },
            indent=2,
        )
    )
    plot_analysis(frame, summary, profiles)


def analyze_toxicity(classifier_device: str) -> None:
    residual_frame, _, _ = compute_analysis()
    frame, summary = compute_toxicity_analysis(residual_frame, classifier_device)
    frame.to_csv(TOXICITY_METRICS_PATH, index=False)
    TOXICITY_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    plot_toxicity(frame, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache-first A-LQR residual smoke test")
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "features",
            "jacobians",
            "controller",
            "rollouts",
            "generate",
            "analyze",
            "toxicity",
            "all",
        ],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--classifier-device", default="cuda:1")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_directories()
    if args.stage == "prepare":
        prepare_prompts()
        return
    if args.stage == "features":
        model, tokenizer = load_model(args.device)
        collect_fit_activations(model, tokenizer)
        return
    if args.stage == "jacobians":
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("shard-index must satisfy 0 <= shard-index < shard-count")
        model, tokenizer = load_model(args.device)
        collect_jacobians(model, tokenizer, args.shard_index, args.shard_count)
        return
    if args.stage == "controller":
        build_controller(args.device)
        return
    if args.stage == "rollouts":
        model, tokenizer = load_model(args.device)
        collect_rollouts(model, tokenizer)
        return
    if args.stage == "generate":
        model, tokenizer = load_model(args.device)
        collect_generations(model, tokenizer)
        return
    if args.stage == "analyze":
        analyze()
        return
    if args.stage == "toxicity":
        analyze_toxicity(args.classifier_device)
        return

    prepare_prompts()
    model, tokenizer = load_model(args.device)
    collect_fit_activations(model, tokenizer)
    collect_jacobians(model, tokenizer, 0, 1)
    del model
    torch.cuda.empty_cache()
    build_controller(args.device)
    model, tokenizer = load_model(args.device)
    collect_rollouts(model, tokenizer)
    collect_generations(model, tokenizer)
    del model
    torch.cuda.empty_cache()
    analyze()
    analyze_toxicity(args.classifier_device)


if __name__ == "__main__":
    main()
