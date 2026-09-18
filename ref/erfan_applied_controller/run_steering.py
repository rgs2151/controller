"""Run steering comparisons for A-LQR, S-PID, and H-infinity.

This script consumes an AppliedControler plan JSON and executes a small
controller-comparison loop on selected model/subset targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


RTP_ID = "allenai/real-toxicity-prompts"
RTP_REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
TOX_CLASSIFIER_ID = "s-nlp/roberta_toxicity_classifier"
TOX_CLASSIFIER_REV = "main"
DEFAULT_SEED = 2151
_TORCH_MODULE = None


@dataclass
class ControllerBuild:
    artifact: ControllerArtifact
    diagnostics: dict[str, object]


def _decoder_layers(model) -> list[object]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise ValueError(
        "Unsupported CausalLM architecture: could not locate decoder layers "
        "(expected model.layers, transformer.h, or gpt_neox.layers)."
    )


def _get_torch():
    global _TORCH_MODULE
    if _TORCH_MODULE is not None:
        return _TORCH_MODULE
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on host runtime.
        raise RuntimeError(
            "Unable to import torch. This environment appears to have a broken "
            "PyTorch/CUDA runtime (for example, missing libcusparseLt.so.0). "
            "Use a working torch environment, or install a CPU-only torch build "
            "for smoke testing."
        ) from exc
    _TORCH_MODULE = torch
    return _TORCH_MODULE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("AppliedControler/experiment_plan_top1.json"),
        help="Plan JSON created by plan_experiments.py",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("AppliedControler/steering_results_top1.csv"),
        help="Output CSV with method-comparison metrics.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("AppliedControler/steering_results_top1.json"),
        help="Output JSON summary.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantized", action="store_true", help="Use 4-bit model loading.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-prompts", type=int, default=12)
    parser.add_argument("--calibration-prompts", type=int, default=16)
    parser.add_argument("--jacobian-prompts", type=int, default=4)
    parser.add_argument("--jacobian-vjp-chunk", type=int, default=256)
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--q-final", type=float, default=1.0)
    parser.add_argument("--setpoint-multiplier", type=float, default=2.0)
    parser.add_argument("--gamma-lower", type=float, default=0.01)
    parser.add_argument("--gamma-upper", type=float, default=10.0)
    parser.add_argument("--gamma-tolerance", type=float, default=1e-4)
    parser.add_argument("--gamma-max-iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--prompt-toxicity-min",
        type=float,
        default=0.0,
        help="Minimum RTP prompt toxicity filter when subset=rtp_random.",
    )
    parser.add_argument(
        "--prompt-toxicity-max",
        type=float,
        default=1.0,
        help="Maximum RTP prompt toxicity filter when subset=rtp_random.",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Optional Hugging Face token. If omitted, HF_TOKEN env/.env is used.",
    )
    parser.add_argument(
        "--limit-experiments",
        type=int,
        default=3,
        help="Maximum number of unique model/subset targets to run.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate plan and runtime dependencies without running steering.",
    )
    return parser.parse_args()


def load_plan(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _method_param(experiment: dict[str, object], key: str, default: float) -> float:
    params = experiment.get("method_params", {})
    if not isinstance(params, dict):
        return default
    value = params.get(key, default)
    return float(value)


def select_unique_targets(plan: dict[str, object], limit: int) -> list[dict[str, object]]:
    experiments = plan.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("plan does not contain an experiments list")
    wanted_methods = {"alqr", "spid", "new_method"}
    chosen: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in experiments:
        if not isinstance(row, dict):
            continue
        method = str(row.get("method", ""))
        if method != "alqr":
            continue
        label = str(row.get("label", ""))
        subset = str(row.get("subset", ""))
        key = (label, subset)
        if key in seen:
            continue
        has_all = all(
            any(
                isinstance(other, dict)
                and str(other.get("label", "")) == label
                and str(other.get("subset", "")) == subset
                and str(other.get("method", "")) == method_name
                for other in experiments
            )
            for method_name in wanted_methods
        )
        if not has_all:
            continue
        chosen.append(row)
        seen.add(key)
        if len(chosen) >= limit:
            break
    return chosen


def benchmark_families_for_subset(subset: str) -> tuple[list[str], str]:
    if subset == "rtp_random":
        return ["rtp"], "id_rtp"
    if subset.startswith("jigsaw_"):
        return ["jigsaw"], "id_rtp"
    if subset.startswith("civil_"):
        return ["civil"], "id_rtp"
    if subset.startswith("toxicchat_"):
        return ["toxicchat"], "id_rtp"
    if subset.startswith("mmlu_"):
        return ["mmlu"], "id_mmlu_high_school_mathematics"
    raise ValueError(f"Unsupported subset for steering run: {subset}")


def pick_subset_records(
    subset: str,
    count: int,
    seed: int,
    prompt_toxicity_min: float,
    prompt_toxicity_max: float,
) -> list[dict[str, object]]:
    if subset == "rtp_random":
        from robust_steerability.benchmarks.toxicity import load_real_toxicity_prompt_pools

        all_records, _, _ = load_real_toxicity_prompt_pools(RTP_ID, RTP_REVISION)
        all_records = [
            row
            for row in all_records
            if prompt_toxicity_min <= float(row["toxicity"]) <= prompt_toxicity_max
        ]
        if not all_records:
            raise ValueError(
                "No RTP prompts after toxicity-range filtering for subset=rtp_random"
            )
        rng = random.Random(seed)
        if len(all_records) <= count:
            return all_records
        return [all_records[idx] for idx in rng.sample(range(len(all_records)), count)]

    from parking.network_size_residual_explore.benchmark_ood_sweep import prepare_prompt_sets

    families, mmlu_id_subject_key = benchmark_families_for_subset(subset)
    mmlu_id_subject = "high_school_mathematics"
    prompt_sets = prepare_prompt_sets(
        num_prompts=max(50, count),
        benchmark_families=families,
        mmlu_id_subject=mmlu_id_subject,
        mmlu_ood_subjects=8,
    )
    if subset not in prompt_sets:
        raise KeyError(f"Subset {subset} not available in prompt sets: {list(prompt_sets.keys())}")
    records = prompt_sets[subset]
    rng = random.Random(seed)
    if len(records) <= count:
        return records
    return [records[idx] for idx in rng.sample(range(len(records)), count)]


def load_model(model_name: str, device: str, quantized: bool, token: str):
    torch = _get_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if quantized:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization,
            torch_dtype=torch.float32,
            device_map={"": int(device.split(":", 1)[1])},
            attn_implementation="eager",
            token=token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, token=token).to(device)
    model.eval()
    return model, tokenizer


def _encode(tokenizer: AutoTokenizer, text: str, model: AutoModelForCausalLM, max_length: int):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    model_device = next(model.parameters()).device
    return {key: value.to(model_device) for key, value in encoded.items()}


def _layer_states_for_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
) -> torch.Tensor:
    torch = _get_torch()
    encoded = _encode(tokenizer, text, model, max_length)
    with torch.no_grad():
        outputs = model(**encoded, use_cache=False, return_dict=True, output_hidden_states=True)
    # hidden_states[1:] aligns with decoder layers.
    stacked = torch.stack([layer[0, -1, :].detach().cpu().float() for layer in outputs.hidden_states[1:]], dim=0)
    return stacked


def _target_from_contrastive_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    toxic_prompts: list[str],
    benign_prompts: list[str],
    max_length: int,
    setpoint_multiplier: float,
) -> dict[str, torch.Tensor]:
    torch = _get_torch()
    toxic_stack = torch.stack([
        _layer_states_for_text(model, tokenizer, text, max_length) for text in toxic_prompts
    ])
    benign_stack = torch.stack([
        _layer_states_for_text(model, tokenizer, text, max_length) for text in benign_prompts
    ])
    toxic_mean = toxic_stack.mean(dim=0)
    benign_mean = benign_stack.mean(dim=0)
    feature = benign_mean - toxic_mean
    feature_norm = torch.linalg.vector_norm(feature, dim=1)
    feature_unit = feature / feature_norm.clamp_min(1e-12).unsqueeze(1)
    return {
        "nominal": benign_mean,
        "feature_unit": feature_unit,
        "feature_norm": feature_norm,
        "beta": setpoint_multiplier * feature_norm,
    }


def _mean_jacobians(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_length: int,
    vjp_chunk_size: int,
) -> torch.Tensor:
    torch = _get_torch()
    from robust_steerability.modeling.jacobians import (
        capture_layer_inputs,
        layer_last_token_jacobian,
    )

    layers = _decoder_layers(model)
    per_prompt = []
    for text in prompts:
        encoded = _encode(tokenizer, text, model, max_length)
        captured = capture_layer_inputs(model, encoded)
        layer_jacs = []
        for layer_index, (hidden, kwargs) in enumerate(captured):
            jac = layer_last_token_jacobian(
                layer=layers[layer_index],
                hidden_states=hidden,
                layer_kwargs=kwargs,
                vjp_chunk_size=vjp_chunk_size,
            )
            layer_jacs.append(jac.float())
        per_prompt.append(torch.stack(layer_jacs, dim=0))
    return torch.stack(per_prompt, dim=0).mean(dim=0)


def build_controllers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    calibration_prompts: int,
    jacobian_prompts: int,
    max_length: int,
    setpoint_multiplier: float,
    q: float,
    r: float,
    q_final: float,
    gamma_lower: float,
    gamma_upper: float,
    gamma_tolerance: float,
    gamma_max_iterations: int,
    seed: int,
    jacobian_vjp_chunk: int,
    device: str,
    include_alqr: bool = True,
    include_hinf: bool = True,
) -> dict[str, ControllerBuild]:
    torch = _get_torch()
    from AppliedControler.method_registry import ControllerArtifact
    from robust_steerability.benchmarks.toxicity import load_real_toxicity_prompt_pools
    from robust_steerability.control import solve_identity_input_lqr

    all_records, toxic_records, nontoxic_records = load_real_toxicity_prompt_pools(RTP_ID, RTP_REVISION)
    del all_records
    rng = random.Random(seed)
    if len(toxic_records) < calibration_prompts or len(nontoxic_records) < calibration_prompts:
        raise ValueError("Not enough RTP calibration prompts to build controllers")
    tox_idx = rng.sample(range(len(toxic_records)), calibration_prompts)
    ben_idx = rng.sample(range(len(nontoxic_records)), calibration_prompts)
    toxic_prompts = [str(toxic_records[idx]["text"]) for idx in tox_idx]
    benign_prompts = [str(nontoxic_records[idx]["text"]) for idx in ben_idx]

    target = _target_from_contrastive_prompts(
        model=model,
        tokenizer=tokenizer,
        toxic_prompts=toxic_prompts,
        benign_prompts=benign_prompts,
        max_length=max_length,
        setpoint_multiplier=setpoint_multiplier,
    )

    dynamics = None
    control_channels = None
    layers = int(target["feature_unit"].shape[0])
    state_dim = int(target["feature_unit"].shape[1])
    if include_alqr or include_hinf:
        jac_source_idx = rng.sample(range(len(nontoxic_records)), jacobian_prompts)
        jac_prompts = [str(nontoxic_records[idx]["text"]) for idx in jac_source_idx]
        dynamics = _mean_jacobians(
            model=model,
            tokenizer=tokenizer,
            prompts=jac_prompts,
            max_length=max_length,
            vjp_chunk_size=jacobian_vjp_chunk,
        )

        layers, state_dim, _ = dynamics.shape
        identity = torch.eye(state_dim, dtype=torch.float32)
        control_channels = identity.unsqueeze(0).repeat(layers, 1, 1)
        state_costs = (q * identity).unsqueeze(0).repeat(layers, 1, 1)
        control_costs = (r * identity).unsqueeze(0).repeat(layers, 1, 1)
        terminal_cost = q_final * identity
    else:
        state_costs = None
        control_costs = None
        terminal_cost = None

    lqr_tracking_gains = None
    if include_alqr:
        if dynamics is None:
            raise RuntimeError("ALQR synthesis requested without dynamics")
        lqr_tracking_gains = solve_identity_input_lqr(
            dynamics=dynamics,
            device=device,
            state_cost=q,
            control_cost=r,
            terminal_cost=q_final,
        )

    builds: dict[str, ControllerBuild] = {
        "spid": ControllerBuild(
            artifact=ControllerArtifact(
                feature_unit=target["feature_unit"],
                setpoints=target["beta"],
                control_channels=control_channels,
            ),
            diagnostics={"controller": "spid", "layers": layers, "state_dim": state_dim},
        ),
    }
    if include_alqr:
        builds["alqr"] = ControllerBuild(
            artifact=ControllerArtifact(
                feature_unit=target["feature_unit"],
                setpoints=target["beta"],
                gains=lqr_tracking_gains,
                control_channels=control_channels,
            ),
            diagnostics={"controller": "alqr", "layers": layers, "state_dim": state_dim},
        )
    if include_hinf:
        from robust_steerability.control import (
            FiniteHorizonControlProblem,
            HInfinityController,
            HInfinityOptions,
        )

        if dynamics is None or control_channels is None or state_costs is None or control_costs is None or terminal_cost is None:
            raise RuntimeError("H-infinity synthesis requested without dynamics/cost tensors")
        problem = FiniteHorizonControlProblem(
            dynamics=dynamics,
            control_channels=control_channels,
            disturbance_channels=control_channels,
            state_costs=state_costs,
            control_costs=control_costs,
            terminal_cost=terminal_cost,
        )
        hinf = HInfinityController.synthesize(
            problem,
            device=device,
            options=HInfinityOptions(
                gamma_lower=gamma_lower,
                gamma_upper=gamma_upper,
                tolerance=gamma_tolerance,
                max_iterations=gamma_max_iterations,
            ),
        )
        hinf_solution = hinf.solution()
        builds["new_method"] = ControllerBuild(
            artifact=ControllerArtifact(
                feature_unit=target["feature_unit"],
                setpoints=target["beta"],
                gains=hinf_solution.gains,
                control_channels=control_channels,
                feasible=hinf_solution.feasible,
                gamma_star=hinf_solution.gamma_star,
                diagnostics=hinf_solution.diagnostics,
            ),
            diagnostics={
                "controller": "new_method",
                "layers": layers,
                "state_dim": state_dim,
                "gamma_star": hinf_solution.gamma_star,
                "feasible": hinf_solution.feasible,
            },
        )
    return builds


def generate_texts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_length: int,
    max_new_tokens: int,
    policy,
    do_sample: bool = True,
) -> list[str]:
    torch = _get_torch()
    from robust_steerability.modeling.interventions import register_generation_policy_hooks

    texts: list[str] = []
    handles = register_generation_policy_hooks(model, policy) if policy is not None else []
    try:
        for prompt in prompts:
            encoded = _encode(tokenizer, prompt, model, max_length)
            with torch.no_grad():
                gen_kwargs = {
                    "repetition_penalty": 1.2,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    gen_kwargs.update({"do_sample": True, "top_p": 0.3, "temperature": 1.0})
                else:
                    gen_kwargs.update({"do_sample": False})
                generated = model.generate(**encoded, **gen_kwargs)
            texts.append(tokenizer.decode(generated[0], skip_special_tokens=True))
    finally:
        for handle in handles:
            handle.remove()
    return texts


def evaluate_methods(
    model,
    tokenizer,
    eval_records: list[dict[str, object]],
    registry,
    controllers: dict[str, ControllerBuild],
    classifier,
    classifier_tokenizer,
    args: argparse.Namespace,
    method_rows: list[dict[str, object]],
    label: str,
    subset: str,
    model_name: str,
) -> list[dict[str, object]]:
    from robust_steerability.benchmarks.toxicity import toxicity_probabilities

    prompts = [str(row["text"]) for row in eval_records]
    baseline_texts = generate_texts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        policy=None,
    )
    baseline_toxicity = toxicity_probabilities(
        baseline_texts,
        model=classifier,
        tokenizer=classifier_tokenizer,
        device=args.device,
    )
    baseline_mean = float(np.mean(baseline_toxicity))

    summaries: list[dict[str, object]] = []
    for row in method_rows:
        method = str(row["method"])
        if method not in {"alqr", "spid", "new_method"}:
            continue
        build = controllers[method]
        feasible = bool(build.artifact.feasible) if build.artifact.feasible is not None else True
        if method == "new_method" and not feasible:
            summaries.append(
                {
                    "label": label,
                    "model_name": model_name,
                    "subset": subset,
                    "method": method,
                    "baseline_toxicity_mean": baseline_mean,
                    "steered_toxicity_mean": baseline_mean,
                    "toxicity_delta": 0.0,
                    "eval_prompts": len(prompts),
                    "max_new_tokens": args.max_new_tokens,
                    "max_length": args.max_length,
                    "run_status": "infeasible",
                    "diagnostics": build.diagnostics,
                }
            )
            continue
        policy = registry[method].build_policy(
            build.artifact,
            kp=_method_param(row, "kp", 1.0),
            ki=_method_param(row, "ki", 0.0),
            kd=_method_param(row, "kd", 0.0),
        )
        run_status = "ok"
        try:
            steered_texts = generate_texts(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                policy=policy,
            )
            steered_toxicity = toxicity_probabilities(
                steered_texts,
                model=classifier,
                tokenizer=classifier_tokenizer,
                device=args.device,
            )
            steered_mean = float(np.mean(steered_toxicity))
        except RuntimeError as exc:
            # Keep long runs alive if one controller is not executable for a target.
            run_status = "runtime_error"
            steered_mean = baseline_mean
            if method == "new_method" and "infeasible H-infinity" in str(exc):
                run_status = "infeasible"
        summaries.append(
            {
                "label": label,
                "model_name": model_name,
                "subset": subset,
                "method": method,
                "baseline_toxicity_mean": baseline_mean,
                "steered_toxicity_mean": steered_mean,
                "toxicity_delta": steered_mean - baseline_mean,
                "eval_prompts": len(prompts),
                "max_new_tokens": args.max_new_tokens,
                "max_length": args.max_length,
                "run_status": run_status,
                "diagnostics": build.diagnostics,
            }
        )
    return summaries


def write_outputs(csv_path: Path, json_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "model_name",
        "subset",
        "method",
        "baseline_toxicity_mean",
        "steered_toxicity_mean",
        "toxicity_delta",
        "eval_prompts",
        "max_new_tokens",
        "max_length",
        "run_status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2) + "\n")


def run_preflight(args: argparse.Namespace) -> int:
    print("Preflight: validating plan and runtime dependencies...")
    if not args.plan.exists():
        print(f"Preflight failed: plan not found at {args.plan}")
        return 2
    plan = load_plan(args.plan)
    targets = select_unique_targets(plan, args.limit_experiments)
    if not targets:
        print("Preflight failed: no targets with full alqr/spid/new_method triplets.")
        return 2
    print(f"Plan check passed: found {len(targets)} runnable target(s).")
    try:
        torch = _get_torch()
    except RuntimeError as exc:
        print(f"Preflight failed: {exc}")
        return 2
    print(f"Torch import passed: version={torch.__version__}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"CUDA available: {cuda_available}")
    if args.device.startswith("cuda") and not cuda_available:
        print(f"Preflight failed: requested device {args.device} but CUDA is unavailable.")
        return 2
    print("Preflight passed.")
    return 0


def main() -> None:
    args = parse_args()
    if args.preflight:
        raise SystemExit(run_preflight(args))

    torch = _get_torch()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but torch.cuda is not available")

    from AppliedControler.method_registry import default_registry
    from robust_steerability.modeling.huggingface import load_access_token, load_sequence_classifier

    repo_root = Path(__file__).resolve().parents[1]
    hf_token = args.hf_token.strip()
    if not hf_token:
        try:
            hf_token = load_access_token(repo_root)
        except RuntimeError:
            hf_token = ""

    plan = load_plan(args.plan)
    targets = select_unique_targets(plan, args.limit_experiments)
    if not targets:
        raise RuntimeError("No runnable targets found in plan for alqr/spid/new_method")

    registry = default_registry()
    classifier, classifier_tokenizer = load_sequence_classifier(
        model_id=TOX_CLASSIFIER_ID,
        revision=TOX_CLASSIFIER_REV,
        device=args.device,
        token=hf_token,
    )

    all_rows: list[dict[str, object]] = []
    experiments = plan.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("plan experiments must be a list")

    for target in targets:
        label = str(target["label"])
        subset = str(target["subset"])
        model_name = str(target["model_name"])
        matching = [
            row for row in experiments
            if isinstance(row, dict)
            and str(row.get("label", "")) == label
            and str(row.get("subset", "")) == subset
            and str(row.get("method", "")) in {"alqr", "spid", "new_method"}
        ]
        if len(matching) < 3:
            continue

        model, tokenizer = load_model(model_name, args.device, args.quantized, token=hf_token)
        controllers = build_controllers(
            model=model,
            tokenizer=tokenizer,
            calibration_prompts=args.calibration_prompts,
            jacobian_prompts=args.jacobian_prompts,
            max_length=args.max_length,
            setpoint_multiplier=args.setpoint_multiplier,
            q=args.q,
            r=args.r,
            q_final=args.q_final,
            gamma_lower=args.gamma_lower,
            gamma_upper=args.gamma_upper,
            gamma_tolerance=args.gamma_tolerance,
            gamma_max_iterations=args.gamma_max_iterations,
            seed=args.seed,
            jacobian_vjp_chunk=args.jacobian_vjp_chunk,
            device=args.device,
        )
        eval_records = pick_subset_records(
            subset,
            args.eval_prompts,
            args.seed,
            args.prompt_toxicity_min,
            args.prompt_toxicity_max,
        )
        rows = evaluate_methods(
            model=model,
            tokenizer=tokenizer,
            eval_records=eval_records,
            registry=registry,
            controllers=controllers,
            classifier=classifier,
            classifier_tokenizer=classifier_tokenizer,
            args=args,
            method_rows=matching,
            label=label,
            subset=subset,
            model_name=model_name,
        )
        all_rows.extend(rows)

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_outputs(args.output_csv, args.output_json, all_rows)
    print(f"Wrote {len(all_rows)} steering rows to {args.output_csv} and {args.output_json}")


if __name__ == "__main__":
    main()
