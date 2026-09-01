from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from scipy.stats import mannwhitneyu, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
METRICS_PATH = PLOTS_DIR / "residual_checks_metrics.csv"
SUMMARY_PATH = PLOTS_DIR / "residual_checks_summary.json"
PLOT_PATH = PLOTS_DIR / "residual_checks.pdf"
PLOT_PREVIEW_PATH = PLOTS_DIR / "residual_checks.png"

MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
RTP_ID = "allenai/real-toxicity-prompts"
RTP_REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
JIGSAW_ID = "tcapelle/jigsaw-toxic-comment-classification-challenge"
JIGSAW_REVISION = "2bf801de1b879f287943ecfc81fdca8690d9fc61"
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
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()

CONDITION_COLORS = {"id": "midnightblue", "ood": "darkred"}
CONDITION_LABELS = {"id": "RTP (ID)", "ood": "Jigsaw (OOD)"}


def load_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("\"'")
                if token:
                    return token
    raise RuntimeError(f"HF_TOKEN is not set and was not found in {env_path}")


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

    rtp = load_dataset(RTP_ID, revision=RTP_REVISION)["train"]
    jigsaw = load_dataset(JIGSAW_ID, revision=JIGSAW_REVISION)["test"]

    toxic: list[dict[str, object]] = []
    nontoxic: list[dict[str, object]] = []
    all_rtp: list[dict[str, object]] = []
    for row_index, row in enumerate(rtp):
        prompt = row["prompt"]
        text = prompt["text"].strip()
        toxicity = prompt["toxicity"]
        if not text or toxicity is None:
            continue
        record = {
            "prompt_id": f"rtp:{row_index}",
            "text": text,
            "source": "rtp",
            "toxicity": float(toxicity),
        }
        if 0.0 <= toxicity <= 1.0:
            all_rtp.append(record)
        if 0.8 <= toxicity <= 1.0:
            toxic.append(record)
        if 0.0 <= toxicity <= 0.1:
            nontoxic.append(record)

    all_jigsaw = [
        {
            "prompt_id": f"jigsaw:{row['id']}",
            "text": row["comment_text"].strip(),
            "source": "jigsaw",
            "toxicity": float(row["toxic"]),
        }
        for row in jigsaw
        if row["comment_text"].strip()
    ]

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


def device_index(device: str) -> int:
    if not device.startswith("cuda:"):
        raise ValueError("The quantized A-LQR model requires a CUDA device such as cuda:0")
    return int(device.split(":", 1)[1])


def load_model(device: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    token = load_hf_token()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=token,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=token,
        quantization_config=quantization,
        dtype=torch.float32,
        device_map={"": device_index(device)},
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def tokenize(tokenizer: AutoTokenizer, text: str, device: torch.device, max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)


def validate_h5(handle: h5py.File, purpose: str) -> None:
    if "config_hash" in handle.attrs and handle.attrs["config_hash"] != CONFIG_HASH:
        raise ValueError(f"Incompatible {purpose} cache")
    if len(handle) > 0 and handle.attrs.get("state_definition") != STATE_DEFINITION:
        raise ValueError(f"Incompatible activation-state definition in {purpose} cache")
    handle.attrs["config_hash"] = CONFIG_HASH
    handle.attrs["model_id"] = MODEL_ID
    handle.attrs["model_revision"] = MODEL_REVISION
    handle.attrs["state_definition"] = STATE_DEFINITION


def write_prompt_group(
    handle: h5py.File,
    group_name: str,
    record: dict[str, object],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    states: torch.Tensor,
    controls: torch.Tensor,
    next_token_id: int,
    next_token_log_probability: float,
) -> None:
    temporary_name = f"tmp_{group_name}"
    if temporary_name in handle:
        del handle[temporary_name]
    group = handle.create_group(temporary_name)
    group.attrs["complete"] = False
    group.attrs["prompt_id"] = str(record["prompt_id"])
    group.attrs["source"] = str(record["source"])
    group.attrs["toxicity"] = float(record["toxicity"])
    group.attrs["text"] = str(record["text"])
    group.attrs["next_token_id"] = next_token_id
    group.attrs["next_token_log_probability"] = next_token_log_probability
    state_array = states.numpy()
    control_array = controls.detach().cpu().float().numpy()
    group.create_dataset(
        "states",
        data=state_array,
        chunks=(1, min(state_array.shape[1], 16), state_array.shape[2]),
        compression="lzf",
        shuffle=True,
    )
    group.create_dataset(
        "controls",
        data=control_array,
        chunks=(1, min(control_array.shape[1], 16), control_array.shape[2]),
        compression="lzf",
        shuffle=True,
    )
    group.create_dataset("input_ids", data=input_ids.detach().cpu().numpy().astype(np.int32))
    group.create_dataset("attention_mask", data=attention_mask.detach().cpu().numpy().astype(np.uint8))
    group.attrs["complete"] = True
    handle.move(temporary_name, group_name)
    handle.file.flush()


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
            write_prompt_group(
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

    feature = means["nontoxic"] - means["toxic"]
    feature_norm = torch.linalg.vector_norm(feature, dim=1)
    feature_unit = feature / feature_norm.clamp_min(EPSILON).unsqueeze(1)
    beta = LAMBDA * feature_norm
    torch.save(
        {
            "config": CONFIG,
            "config_hash": CONFIG_HASH,
            "state_definition": STATE_DEFINITION,
            "nominal": means["nontoxic"],
            "toxic_mean": means["toxic"],
            "feature": feature,
            "feature_norm": feature_norm,
            "feature_unit": feature_unit,
            "beta": beta,
        },
        FEATURES_PATH,
    )


def capture_layer_inputs(
    model: AutoModelForCausalLM,
    encoded: dict[str, torch.Tensor],
) -> list[tuple[torch.Tensor, dict[str, object]]]:
    captured: list[tuple[torch.Tensor, dict[str, object]] | None] = [None] * len(model.model.layers)
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, args, kwargs):
            kept_kwargs: dict[str, object] = {}
            for name in ["attention_mask", "position_ids", "cache_position", "position_embeddings"]:
                value = kwargs.get(name)
                if isinstance(value, tuple):
                    kept_kwargs[name] = tuple(item.detach() for item in value)
                elif torch.is_tensor(value):
                    kept_kwargs[name] = value.detach()
                elif value is not None:
                    kept_kwargs[name] = value
            kept_kwargs["use_cache"] = False
            captured[layer_index] = (args[0].detach(), kept_kwargs)

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(make_hook(layer_index), with_kwargs=True))
    with torch.no_grad():
        model(**encoded, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    if any(item is None for item in captured):
        raise RuntimeError("Failed to capture all transformer layer inputs")
    return [item for item in captured if item is not None]


def layer_last_token_jacobian(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    layer_kwargs: dict[str, object],
) -> torch.Tensor:
    prefix = hidden_states[:, :-1, :].detach()
    last_state = hidden_states[0, -1, :].detach().requires_grad_(True)

    def block_last(state: torch.Tensor) -> torch.Tensor:
        full_state = torch.cat([prefix, state.view(1, 1, -1)], dim=1)
        output = layer(full_state, **layer_kwargs)
        if isinstance(output, tuple):
            output = output[0]
        return output[0, -1, :]

    output = block_last(last_state)
    dimension = output.numel()
    rows = []
    for start in range(0, dimension, JACOBIAN_VJP_CHUNK):
        stop = min(start + JACOBIAN_VJP_CHUNK, dimension)
        grad_outputs = torch.zeros(stop - start, dimension, device=output.device, dtype=output.dtype)
        grad_outputs[torch.arange(stop - start, device=output.device), torch.arange(start, stop, device=output.device)] = 1
        gradient = torch.autograd.grad(
            output,
            last_state,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=stop < dimension,
        )[0]
        rows.append(gradient.detach().cpu())
    return torch.cat(rows, dim=0)


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
    dynamics = jacobians.to(device=device, dtype=torch.float32)
    layer_count, dimension, _ = dynamics.shape
    identity = torch.eye(dimension, device=device, dtype=torch.float32)
    value = Q_FINAL * identity
    gains = torch.empty_like(dynamics)
    for layer_index in reversed(range(layer_count)):
        layer_dynamics = dynamics[layer_index]
        lhs = value + R * identity
        rhs = value @ layer_dynamics
        gain = torch.linalg.solve(lhs, rhs)
        state_cost = Q * identity + layer_dynamics.T @ value @ layer_dynamics
        value = state_cost - rhs.T @ gain
        value = 0.5 * (value + value.T)
        gains[layer_index] = gain
        print(f"Riccati layer {layer_index + 1}/{layer_count}", flush=True)
    return gains.cpu()


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
) -> tuple[object, torch.Tensor, torch.Tensor]:
    layer_count = len(model.model.layers)
    sequence_length = int(encoded["input_ids"].shape[1])
    hidden_size = model.config.hidden_size
    controls = torch.zeros(layer_count, sequence_length, hidden_size, dtype=torch.float32)
    states: list[torch.Tensor | None] = [None] * (layer_count + 1)
    handles = []

    def make_input_hook(layer_index: int):
        def hook(_module, args):
            states[layer_index] = args[0][0].detach().cpu().float()

        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(make_input_hook(layer_index)))

    if controller is not None:
        gains = controller["gains"].to(model.device)
        feature_unit = controller["feature_unit"].to(model.device)
        beta = controller["beta"].to(model.device)

        def make_hook(layer_index: int):
            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activation = _args[0][:, -1, :]
                alpha = beta[layer_index] - activation @ feature_unit[layer_index]
                error = alpha.unsqueeze(1) * feature_unit[layer_index].unsqueeze(0)
                control = error @ gains[layer_index].T
                changed = hidden.clone()
                changed[:, -1, :] = changed[:, -1, :] + control.to(changed.dtype)
                controls[layer_index, -1, :] = control[0].detach().cpu().float()
                if layer_index == layer_count - 1:
                    states[layer_count] = changed[0].detach().cpu().float()
                if isinstance(output, tuple):
                    return (changed,) + output[1:]
                return changed

            return hook

        for layer_index, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(make_hook(layer_index)))
    else:
        def final_output_hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            states[layer_count] = hidden[0].detach().cpu().float()

        handles.append(model.model.layers[-1].register_forward_hook(final_output_hook))
    with torch.no_grad():
        output = model(
            **encoded,
            use_cache=False,
            return_dict=True,
        )
    for handle in handles:
        handle.remove()
    if any(state is None for state in states):
        raise RuntimeError("Failed to cache every raw transformer state")
    return output, torch.stack([state for state in states if state is not None]), controls


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
            write_prompt_group(
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


def read_rollout(path: Path) -> list[dict[str, object]]:
    records = []
    with h5py.File(path, "r") as handle:
        if handle.attrs["config_hash"] != CONFIG_HASH:
            raise ValueError(f"Incompatible rollout cache: {path}")
        for name in sorted(handle.keys()):
            if name.startswith("tmp_"):
                continue
            group = handle[name]
            records.append(
                {
                    "prompt_id": group.attrs["prompt_id"],
                    "states": torch.from_numpy(group["states"][:]).float(),
                    "controls": torch.from_numpy(group["controls"][:]).float(),
                }
            )
    return records


def residual_metrics(
    states: torch.Tensor,
    controls: torch.Tensor,
    controller: dict[str, object],
) -> dict[str, object]:
    nominal = controller["nominal"].float()
    dynamics = controller["jacobians"].float()
    gains = controller["gains"].float()
    feature_unit = controller["feature_unit"].float()
    feature_norm = controller["feature_norm"].float()
    beta = controller["beta"].float()
    last_states = states[:, -1, :]
    last_controls = controls[:, -1, :]
    deviations = last_states[:-1] - nominal[:-1]
    predicted = nominal[1:] + torch.einsum("lij,lj->li", dynamics, deviations) + last_controls
    residuals = last_states[1:] - predicted
    residual_magnitude = float(torch.linalg.vector_norm(residuals).item())

    residual_response = torch.zeros_like(last_states[0])
    for layer_index in range(dynamics.shape[0]):
        gain_feature = gains[layer_index] @ feature_unit[layer_index]
        residual_response = (
            dynamics[layer_index] @ residual_response
            - gain_feature * torch.dot(feature_unit[layer_index], residual_response)
            + residuals[layer_index]
        )
    final_scale = float(feature_norm[-1].clamp_min(EPSILON).item())
    directional_effect = abs(float(torch.dot(feature_unit[-1], residual_response).item())) / final_scale
    amplification = directional_effect / max(residual_magnitude, EPSILON)
    failure = abs(float((beta[-1] - torch.dot(feature_unit[-1], last_states[-1])).item())) / final_scale
    layer_relative_residual = torch.linalg.vector_norm(residuals, dim=1) / torch.linalg.vector_norm(
        last_states[1:], dim=1
    ).clamp_min(EPSILON)
    return {
        "residual_magnitude": residual_magnitude,
        "directional_effect": directional_effect,
        "amplification": amplification,
        "failure": failure,
        "layer_relative_residual": layer_relative_residual.numpy(),
    }


def bootstrap_correlation_difference(frame: pd.DataFrame) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, len(frame), len(frame))
        sample = frame.iloc[indices]
        directional_rho = spearmanr(sample["directional_effect"], sample["failure"]).statistic
        magnitude_rho = spearmanr(sample["residual_magnitude"], sample["failure"]).statistic
        values.append(float(directional_rho - magnitude_rho))
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
    correlation_difference_interval = bootstrap_correlation_difference(frame)
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
        "mann_whitney_ood_greater_than_id": {
            "u": float(amplification_test.statistic),
            "p_value": float(amplification_test.pvalue),
        },
        "empirical_gamma_95": {
            "id": float(id_amplification.quantile(0.95)),
            "ood": float(ood_amplification.quantile(0.95)),
        },
    }
    return frame, summary, layer_profiles


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


def plot_analysis(frame: pd.DataFrame, summary: dict[str, object], profiles: dict[str, np.ndarray]) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))
    depth = np.linspace(0, 1, len(profiles["id_alqr"]))
    for condition in ["id", "ood"]:
        axes[0, 0].plot(
            depth,
            profiles[f"{condition}_baseline"],
            color=CONDITION_COLORS[condition],
            linestyle=":",
            label=f"{CONDITION_LABELS[condition]}, baseline",
        )
        axes[0, 0].plot(
            depth,
            profiles[f"{condition}_alqr"],
            color=CONDITION_COLORS[condition],
            linestyle="-",
            label=f"{CONDITION_LABELS[condition]}, A-LQR",
        )
    axes[0, 0].set_title("A  Layer residuals")
    axes[0, 0].set_xlabel("Normalized depth")
    axes[0, 0].set_ylabel(r"Mean $\|\xi_k\|_2/\|x_{k+1}\|_2$")
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_xticks([0, 1])
    profile_values = np.concatenate(list(profiles.values()))
    axes[0, 0].set_ylim(0, float(profile_values.max()) * 1.05)
    axes[0, 0].set_yticks([0, float(profile_values.max())])
    axes[0, 0].legend(loc="best", fontsize=7)

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
        ax=axes[0, 1],
    )
    medians = frame.groupby("condition")["amplification"].median()
    for position, condition in enumerate(["id", "ood"]):
        axes[0, 1].plot([position - 0.22, position + 0.22], [medians[condition]] * 2, color="black")
    axes[0, 1].set_title("B  Residual amplification")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(["RTP\n(ID)", "Jigsaw\n(OOD)"])
    axes[0, 1].set_ylabel(r"$|T_{\rm LQR}\xi|/\|\xi\|_2$")
    axes[0, 1].set_ylim(0, float(frame["amplification"].max()) * 1.05)
    axes[0, 1].set_yticks([0, float(frame["amplification"].max())])

    for condition in ["id", "ood"]:
        subset = frame[frame["condition"] == condition]
        axes[1, 0].scatter(
            np.log10(subset["residual_magnitude"] + EPSILON),
            subset["failure"],
            color=CONDITION_COLORS[condition],
            s=16,
            alpha=0.7,
            label=CONDITION_LABELS[condition],
        )
        axes[1, 1].scatter(
            np.log10(subset["directional_effect"] + EPSILON),
            subset["failure"],
            color=CONDITION_COLORS[condition],
            s=16,
            alpha=0.7,
        )
    magnitude_rho = summary["spearman_residual_magnitude_vs_failure"]["rho"]
    directional_rho = summary["spearman_directional_effect_vs_failure"]["rho"]
    axes[1, 0].set_title(rf"C  Magnitude, $\rho={magnitude_rho:.2f}$")
    axes[1, 0].set_xlabel(r"$\log_{10}\|\xi\|_2$")
    axes[1, 0].set_ylabel("Final normalized tracking error")
    axes[1, 0].legend(loc="best", fontsize=7)
    axes[1, 1].set_title(rf"D  Direction-aware effect, $\rho={directional_rho:.2f}$")
    axes[1, 1].set_xlabel(r"$\log_{10}|T_{\rm LQR}\xi|$")
    axes[1, 1].set_ylabel("Final normalized tracking error")

    endpoint_ticks(axes[1, 0], np.log10(frame["residual_magnitude"].to_numpy() + EPSILON), "x")
    endpoint_ticks(axes[1, 1], np.log10(frame["directional_effect"].to_numpy() + EPSILON), "x")
    failure_max = float(frame["failure"].max())
    for axis in [axes[1, 0], axes[1, 1]]:
        axis.set_ylim(0, failure_max * 1.05)
        axis.set_yticks([0, failure_max])

    for axis in axes.flat:
        axis.set_box_aspect(1)
        sns.despine(ax=axis, trim=True, offset=10)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.94, wspace=0.48, hspace=0.55)
    fig.savefig(PLOT_PATH, bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(PLOT_PREVIEW_PATH, bbox_inches="tight", facecolor="white", transparent=False, dpi=220)
    plt.close(fig)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache-first A-LQR residual smoke test")
    parser.add_argument(
        "stage",
        choices=["prepare", "features", "jacobians", "controller", "rollouts", "analyze", "all"],
    )
    parser.add_argument("--device", default="cuda:0")
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
    if args.stage == "analyze":
        analyze()
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
    del model
    torch.cuda.empty_cache()
    analyze()


if __name__ == "__main__":
    main()
