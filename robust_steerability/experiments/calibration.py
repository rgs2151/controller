"""Historical combined calibration used by ``ref/paper_benchmark_50``.

Source-faithful non-H-infinity calibration now lives in
``robust_steerability.source_methods.calibration``. This module stays in place
to preserve the completed pilot and Hannah's existing H-infinity handoff.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict
from pathlib import Path

import torch

from robust_steerability.artifacts import configuration_hash, implementation_hash
from robust_steerability.benchmarks.calibration import calibration_records
from robust_steerability.modeling.interventions import _decoder_layers
from robust_steerability.calibration.disturbances import fit_disturbance_geometry
from robust_steerability.calibration.nominal import average_prompt_jacobians, project_dynamics
from robust_steerability.calibration.targets import build_contrastive_target
from robust_steerability.control.lqr import solve_identity_input_lqr
from robust_steerability.control import (
    FiniteHorizonControlProblem,
    HInfinityController,
    HInfinityOptions,
)
from robust_steerability.experiments.methods import ControllerArtifact
from robust_steerability.experiments.baselines import fit_baselines
from robust_steerability.experiments.diagnostics import cpu_tensors, score, verify_run


def collect_last_token_states(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Collect decoder input/output states as ``(records, L+1, hidden)``."""

    batches = []
    head_batches = []
    layers = _decoder_layers(model)
    model_device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model_device)
        states = [None] * (len(layers) + 1)
        heads = [None] * len(layers)
        handles = []
        def head_hook(index):
            def capture(_module, args):
                heads[index] = args[0][:, -1, :].detach().float()
            return capture
        def input_hook(index):
            def capture(_module, args):
                states[index] = args[0][:, -1, :].detach().float()
            return capture
        def terminal_hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            states[-1] = hidden[:, -1, :].detach().float()
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(input_hook(index)))
            if model.config.model_type == "gpt2":
                projection = layer.attn.c_proj
            elif model.config.model_type in {"llama", "qwen2", "mistral", "gemma", "gemma2"}:
                projection = layer.self_attn.o_proj
            else:
                raise ValueError(f"Attention-head capture unsupported for {model.config.model_type}")
            handles.append(projection.register_forward_pre_hook(head_hook(index)))
        handles.append(layers[-1].register_forward_hook(terminal_hook))
        try:
            with torch.inference_mode():
                model(**encoded, return_dict=True, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        batches.append(torch.stack(states, dim=1).cpu())
        head_batches.append(torch.stack(heads, dim=1).cpu())
    return {"hidden": torch.cat(batches, dim=0), "attention_heads": torch.cat(head_batches, dim=0)}


def _fit_reduced_basis(fit_states, rank, epsilon, contrast):
    """Fit an orthonormal target-preserving basis from fit states only."""
    means = fit_states.mean(dim=0)
    retained_rank = min(rank, fit_states.shape[0] - 2, fit_states.shape[2])
    bases = []
    for k in range(fit_states.shape[1]):
        direction = contrast[k] / contrast[k].norm().clamp_min(epsilon)
        centered = fit_states[:, k] - means[k]
        orthogonal = centered - (centered @ direction).unsqueeze(1) * direction
        _, _, vh = torch.linalg.svd(orthogonal, full_matrices=False)
        basis = torch.linalg.qr(
            torch.cat([direction[:, None], vh[:retained_rank - 1].T], dim=1)
        ).Q
        alignment = torch.sign(basis[:, 0] @ direction)
        basis[:, 0] *= torch.where(alignment == 0, alignment.new_tensor(1), alignment)
        bases.append(basis)
    return means, torch.stack(bases)


def _apply_coordinates(
    states: torch.Tensor,
    means: torch.Tensor,
    encoders: torch.Tensor,
) -> torch.Tensor:
    centered = states - means.unsqueeze(0)
    return torch.einsum("nld,ldr->nlr", centered, encoders)


def _fit_controller_inputs(model, tokenizer, settings: dict, cache_path: Path) -> dict:
    """Identify actual full-order and projected Jacobian dynamics."""
    seed = int(settings["seed"])
    fit_per_class = int(settings["fit_prompts_per_class"])
    calibration_count = int(settings["disturbance_prompts"])
    negative, positive, calibration_records_list, dataset = calibration_records(
        settings["behavior"], fit_per_class, calibration_count, seed)
    fit_records = negative + positive
    jacobian_records = positive[:int(settings["jacobian_prompts"])]
    if len(jacobian_records) != int(settings["jacobian_prompts"]):
        raise ValueError("Not enough positive fit prompts for Jacobian identification")
    raw_dynamics = average_prompt_jacobians(
        model, tokenizer, jacobian_records, cache_dir=cache_path.parent / (cache_path.stem + "_jacobians"),
        max_length=int(settings["jacobian_max_length"]), vjp_chunk_size=int(settings["jacobian_vjp_chunk_size"]),
        model_revision=str(settings["model_loading"]["revision"]))
    fit_states = collect_last_token_states(
        model,
        tokenizer,
        [str(row["text"]) for row in fit_records],
        max_length=int(settings["calibration_max_length"]),
        batch_size=int(settings["activation_batch_size"]),
    )
    calibration_states = collect_last_token_states(
        model,
        tokenizer,
        [str(row["text"]) for row in calibration_records_list],
        max_length=int(settings["calibration_max_length"]),
        batch_size=int(settings["activation_batch_size"]),
    )
    device = next(model.parameters()).device
    fit_heads, calibration_heads = fit_states["attention_heads"], calibration_states["attention_heads"]
    fit_states = fit_states["hidden"].to(device)
    calibration_states = calibration_states["hidden"].to(device)
    raw_contrast = fit_states[fit_per_class:].mean(dim=0) - fit_states[:fit_per_class].mean(dim=0)
    raw_target = build_contrastive_target(
        fit_states[fit_per_class:].mean(dim=0), fit_states[:fit_per_class].mean(dim=0),
        float(settings["hinf_setpoint_multiplier"]))
    means, basis = _fit_reduced_basis(
        fit_states, int(settings["state_rank"]), float(settings["numerical_floor"]), raw_contrast)
    encoders_all = decoders_all = basis
    fit_reduced = _apply_coordinates(fit_states, means, encoders_all)
    calibration_reduced = _apply_coordinates(calibration_states, means, encoders_all)
    dynamics = project_dynamics(raw_dynamics, encoders_all, decoders_all)
    feature_unit_all = torch.einsum("ldr,ld->lr", basis, raw_target["feature_unit"])
    feature_unit_all = feature_unit_all / feature_unit_all.norm(dim=1, keepdim=True).clamp_min(
        float(settings["numerical_floor"]))
    setpoints_all = raw_target["beta"] - torch.einsum("ld,ld->l", means, raw_target["feature_unit"])

    def semantic_deviation(reduced):
        scalar = torch.einsum("nlr,lr->nl", reduced, feature_unit_all) - setpoints_all
        return scalar.unsqueeze(-1) * feature_unit_all.unsqueeze(0)

    fit_deviations = semantic_deviation(fit_reduced)
    calibration_deviations = semantic_deviation(calibration_reduced)
    residuals = calibration_deviations[:, 1:] - torch.einsum(
        "lij,nlj->nli", dynamics, calibration_deviations[:, :-1])
    fit_residuals = fit_deviations[:, 1:] - torch.einsum(
        "lij,nlj->nli", dynamics, fit_deviations[:, :-1])
    disturbance = fit_disturbance_geometry(
        fit_residuals,
        variance_threshold=float(settings["disturbance_variance"]),
    )

    horizon, state_dimension, _ = dynamics.shape
    identity = torch.eye(state_dimension, device=device)
    control_channels = identity.unsqueeze(0).repeat(horizon, 1, 1)
    depth_weight = 1.0 / horizon
    reference_states = torch.zeros(horizon + 1, state_dimension, device=device)
    reference_controls = torch.zeros(horizon, state_dimension, device=device)
    readout_bases = torch.linalg.qr(torch.cat([
        feature_unit_all.unsqueeze(-1), identity.expand(horizon + 1, -1, -1)], dim=-1)).Q
    readouts = readout_bases.transpose(-1, -2)
    output_std = torch.einsum("lpr,nlr->nlp", readouts, calibration_reduced).std(dim=0, correction=1)
    output_std = output_std.clamp_min(float(settings["numerical_floor"]))
    normalized_readouts = readouts / output_std.unsqueeze(-1)
    cost_metric = normalized_readouts.transpose(-1, -2) @ normalized_readouts
    state_costs = float(settings["q"]) * depth_weight * cost_metric[:-1]
    coordinates = torch.einsum(
        "lri,nli->nlr", torch.linalg.pinv(disturbance.channels), residuals - disturbance.means)
    energies = coordinates.square().sum(dim=(1, 2)).sqrt()
    coverage_scale = torch.quantile(
        energies, float(settings["disturbance_coverage"]), interpolation="higher")
    if not torch.isfinite(coverage_scale) or coverage_scale <= 0:
        raise ValueError("Disturbance scaling energy must be positive")
    scaled_channels = coverage_scale * disturbance.channels
    remainder = residuals - disturbance.means - torch.einsum("lir,nlr->nli", disturbance.channels, coordinates)
    control_costs = (float(settings["r"]) * depth_weight * identity).unsqueeze(0).repeat(
        horizon, 1, 1
    )
    terminal_cost = float(settings["q_final"]) * cost_metric[-1]


    problem = FiniteHorizonControlProblem(
        dynamics=dynamics, control_channels=control_channels,
        disturbance_channels=scaled_channels, state_costs=state_costs,
        control_costs=control_costs, terminal_cost=terminal_cost,
    )
    return {
        "raw_dynamics": raw_dynamics,
        "raw_target": raw_target,
        "problem": asdict(problem),
        "maps": {"means": means, "encoders": encoders_all, "decoders": decoders_all,
                 "feature_unit": feature_unit_all, "setpoints": setpoints_all},
        "splits": {"fit": [str(row["prompt_id"]) for row in fit_records],
                   "calibration": [str(row["prompt_id"]) for row in calibration_records_list]},
        "normalization": {
            "protocol_id": "orthonormal-reduced-readout-depth-v1", "coordinates": "raw",
            "state_whitening": identity.unsqueeze(0).repeat(horizon + 1, 1, 1),
            "control_std": torch.ones(horizon, state_dimension, device=device),
            "semantic_output_std": output_std[:, :1],
            "depth_increment": torch.full((horizon,), depth_weight),
            "stage_costs_depth_weighted": True,
            "description": "Fit-only target-preserving orthonormal basis; no state whitening; "
                           "standardized performance readouts; orthonormal intervention channels; Q and R divided by T.",
        },
        "calibration": {
            "source_snapshots": {name: Path(inspect.getfile(obj)).read_text() for name, obj in
                                 (("calibration.py", calibrate_controller), ("baselines.py", fit_baselines))},
            "dynamics_estimator": "averaged last-token transformer Jacobians, with prefix states fixed; projected using next-layer encoders and current-layer decoders",
            "jacobian_prompt_ids": [row["prompt_id"] for row in jacobian_records],
            "raw_jacobians": raw_dynamics,
            "reference_rule": "nearest point on the semantic setpoint hyperplane, recomputed from each current context",
            "residuals": residuals, "state_basis": basis,
            "means": means, "encoders": encoders_all, "decoders": decoders_all,
            "fit_reduced_states": fit_reduced, "calibration_reduced_states": calibration_reduced,
            "fit_state_deviations": fit_deviations,
            "calibration_state_deviations": calibration_deviations,
            "fit_residuals": fit_residuals,
            "fit_labels": [0] * fit_per_class + [1] * fit_per_class,
            "fit_records": fit_records, "calibration_records": calibration_records_list,
            "fit_hidden_states": fit_states, "calibration_hidden_states": calibration_states,
            "fit_attention_heads": fit_heads, "calibration_attention_heads": calibration_heads,
            "attention_head_count": int(model.config.num_attention_heads),
            "target_readouts": normalized_readouts[:, :1],
            "protected_readouts": normalized_readouts[:, 1:],
            "protected_output_std": output_std[:, 1:],
            "reference_states": reference_states,
            "reference_controls": reference_controls,
            "setpoints": setpoints_all,
            "feedback_definition": "context-updated semantic tracking error; intervention = K @ feedback",
            "protected_readouts_definition": "Orthogonal complement of the target within the reduced basis; representation preservation, not an independent behavior probe.",
            "reference_controls_definition": "zero nominal feedforward; feedback alone supplies the intervention",
            "disturbance_construction": {
                "method": "fit-only centered PCA covariance factor, ddof=1, scaled by calibration trajectory-energy quantile",
                "unscaled_channels": disturbance.channels,
                "coverage_scale": coverage_scale,
                "coverage_quantile": settings["disturbance_coverage"],
                "calibration_energy": energies,
                "calibration_coordinates": coordinates,
                "out_of_subspace_residuals": remainder,
                "variance_threshold": float(settings["disturbance_variance"]),
                "means": disturbance.means,
                "retained_ranks": disturbance.retained_ranks,
                "explained_variance": disturbance.explained_variance,
            },
            "dataset": dataset,
            "settings": settings,
        },
    }


def _options(settings: dict) -> HInfinityOptions:
    return HInfinityOptions(
        gamma_lower=float(settings["gamma_lower"]), gamma_upper=float(settings["gamma_upper"]),
        tolerance=float(settings["gamma_tolerance"]), max_iterations=int(settings["gamma_max_iterations"]),
        deployment_margin=float(settings["gamma_deployment_margin"]),
    )


def _fingerprint(model_label: str, model_id: str, settings: dict) -> str:
    return configuration_hash({
        "model_label": model_label, "model_id": model_id, "settings": settings,
        "implementation_sha256": implementation_hash(),
    })


def diagnostic_root(cache_path: Path) -> Path:
    return cache_path.parent / (cache_path.stem + "_diagnostics")


def diagnostic_run(cache_path: Path, fingerprint: str) -> Path:
    return diagnostic_root(cache_path) / "runs" / ("calibration-" + fingerprint[:20])


def _freeze_inputs(inputs: dict, solution, *, model, model_label: str, model_id: str,
                   settings: dict, fingerprint: str, cache_path: Path, device: str) -> Path:
    protocol_settings = {key: value for key, value in settings.items() if key != "model_loading"}
    bundle = {
        "problem": inputs["problem"], "options": asdict(_options(settings)),
        "record": {
            "run_id": "calibration-" + fingerprint[:20], "model_id": model_id,
            "model_label": model_label, "model_revision": settings["model_loading"]["revision"],
            "model_family": str(model.config.model_type), "parameter_count": int(model.num_parameters()),
            "behavior": settings["behavior"], "intervention_channel": "reduced_semantic_setpoint",
            "protocol_id": "context-semantic-" + configuration_hash(protocol_settings)[:16],
            "synthetic": False, "calibration_fingerprint": fingerprint,
        },
        "splits": inputs["splits"], "normalization": inputs["normalization"],
        "calibration": inputs["calibration"],
        # Do not substitute unvalidated approximations for Hannah's baseline predictors.
        "predictors": {},
    }
    root = diagnostic_root(cache_path)
    root.mkdir(parents=True, exist_ok=True)
    bundle_path = root / "input.pt"
    torch.save(cpu_tensors(bundle), bundle_path)
    return score(bundle_path, device, cache_root=root, solution=solution)


def calibrate_controller(
    model, tokenizer, *, model_label: str, model_id: str, cache_path: Path,
    settings: dict[str, object], controller_device: str,
) -> tuple[ControllerArtifact, dict[str, object]]:
    """Fit/load the shared controller; new fits freeze full H-infinity diagnostics."""
    fingerprint = _fingerprint(model_label, model_id, settings)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached["fingerprint"] != fingerprint:
            raise ValueError(f"Incompatible controller cache: {cache_path}")
        verify_run(diagnostic_run(cache_path, fingerprint))
        return ControllerArtifact(**cached["artifact"]), cached["metadata"]

    inputs = _fit_controller_inputs(model, tokenizer, settings, cache_path)
    problem = FiniteHorizonControlProblem(**inputs["problem"])
    lqr_gains = -solve_identity_input_lqr(inputs["raw_dynamics"], controller_device,
                                        float(settings["alqr_q"]), float(settings["alqr_r"]),
                                        float(settings["alqr_q_final"]))
    hinf = HInfinityController.synthesize(
        problem, device=controller_device, options=_options(settings),
    ).solution()
    maps = inputs["maps"]
    artifact = ControllerArtifact(
        means=maps["means"][:-1], encoders=maps["encoders"][:-1], decoders=maps["decoders"][1:],
        feature_unit=maps["feature_unit"][:-1], setpoints=maps["setpoints"][:-1],
        control_channels=problem.control_channels, lqr_gains=lqr_gains,
        hinf_gains=hinf.gains, hinf_feasible=hinf.feasible,
        gamma_star=hinf.gamma_star, hinf_diagnostics=hinf.diagnostics,
        raw_feature_unit=inputs["raw_target"]["feature_unit"][:-1],
        alqr_setpoints=(float(settings["alqr_setpoint_multiplier"]) *
                        inputs["raw_target"]["feature_norm"][:-1]),
        spid_setpoints=(float(settings["spid_setpoint_multiplier"]) *
                       inputs["raw_target"]["feature_norm"][:-1]),
        baselines=fit_baselines(inputs["calibration"], seed=int(settings["seed"]),
                                strengths=settings["baseline_strengths"]),
    )
    disturbance = inputs["calibration"]["disturbance_construction"]
    metadata = {
        "behavior": settings["behavior"],
        "source_fit_prompt_ids": sorted({row.get("source_prompt_id", row["prompt_id"]) for row in inputs["calibration"]["fit_records"]}),
        "fingerprint": fingerprint, "model_label": model_label, "model_id": model_id,
        "fit_prompt_ids": sorted(inputs["splits"]["fit"]),
        "calibration_prompt_ids": inputs["splits"]["calibration"],
        "tuning_records": inputs["calibration"]["calibration_records"],
        "state_rank": problem.state_dimension, "horizon": problem.horizon,
        "disturbance_ranks": disturbance["retained_ranks"].tolist(),
        "disturbance_explained_variance": disturbance["explained_variance"].tolist(),
        "gamma_star": hinf.gamma_star,
        "robust_steerability": None if hinf.gamma_star is None else 1.0 / hinf.gamma_star,
        "hinf_feasible": hinf.feasible,
    }
    inputs["calibration"]["baseline_parameters"] = artifact.baselines
    inputs["calibration"]["lqr_solution"] = {
        "controller": "alqr", "gains": lqr_gains,
        "feedback_definition": "u = K @ ((h dot v - beta) * v); post-block addition, identity input channel",
        "costs": {key: settings[key] for key in ("alqr_q", "alqr_r", "alqr_q_final")},
        "state_coordinates": "full physical hidden state",
    }
    inputs["calibration"]["pid_gains"] = {key: settings[key] for key in ("kp", "ki", "kd")}
    _freeze_inputs(inputs, hinf, model=model, model_label=model_label, model_id=model_id,
                   settings=settings, fingerprint=fingerprint, cache_path=cache_path,
                   device=controller_device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".pt.tmp")
    torch.save(cpu_tensors({"fingerprint": fingerprint, "artifact": artifact.__dict__,
                           "metadata": metadata}), temporary)
    temporary.replace(cache_path)
    return artifact, metadata
