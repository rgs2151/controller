"""Cache-first calibration of reduced transformer control problems."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.benchmarks.ood import RTP_ID, RTP_REVISION, stable_sample
from robust_steerability.benchmarks.toxicity import load_real_toxicity_prompt_pools
from robust_steerability.calibration.disturbances import fit_disturbance_geometry
from robust_steerability.control import (
    FiniteHorizonControlProblem,
    HInfinityController,
    HInfinityOptions,
    solve_identity_input_lqr,
)
from robust_steerability.experiments.methods import ReducedControllerArtifact


def collect_last_token_states(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    """Collect decoder input/output states as ``(records, L+1, hidden)``."""

    batches = []
    model_device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model_device)
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        layer_states = torch.stack(
            [state[:, -1, :].detach().cpu().float() for state in output.hidden_states],
            dim=1,
        )
        batches.append(layer_states)
    return torch.cat(batches, dim=0)


def _fit_reduced_coordinates(
    fit_states: torch.Tensor,
    rank: int,
    scale_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layer_count = fit_states.shape[1]
    hidden_size = fit_states.shape[2]
    retained_rank = min(rank, fit_states.shape[0] - 1, hidden_size)
    means = torch.empty(layer_count, hidden_size)
    encoders = torch.empty(layer_count, hidden_size, retained_rank)
    decoders = torch.empty_like(encoders)
    reduced = torch.empty(fit_states.shape[0], layer_count, retained_rank)
    for layer_index in range(layer_count):
        values = fit_states[:, layer_index]
        mean = values.mean(dim=0)
        centered = values - mean
        _, singular_values, right_vectors = torch.linalg.svd(
            centered,
            full_matrices=False,
        )
        basis = right_vectors[:retained_rank].T.contiguous()
        scales = (
            singular_values[:retained_rank] / max(1, values.shape[0] - 1) ** 0.5
        ).clamp_min(scale_floor)
        means[layer_index] = mean
        encoders[layer_index] = basis / scales
        decoders[layer_index] = basis * scales
        reduced[:, layer_index] = centered @ encoders[layer_index]
    return means, encoders, decoders, reduced


def _apply_coordinates(
    states: torch.Tensor,
    means: torch.Tensor,
    encoders: torch.Tensor,
) -> torch.Tensor:
    centered = states - means.unsqueeze(0)
    return torch.einsum("nld,ldr->nlr", centered, encoders)


def _fit_dynamics(reduced: torch.Tensor, ridge: float) -> torch.Tensor:
    horizon = reduced.shape[1] - 1
    state_dimension = reduced.shape[2]
    identity = torch.eye(state_dimension)
    dynamics = []
    for layer_index in range(horizon):
        inputs = reduced[:, layer_index]
        outputs = reduced[:, layer_index + 1]
        coefficients = torch.linalg.solve(
            inputs.T @ inputs + ridge * identity,
            inputs.T @ outputs,
        )
        dynamics.append(coefficients.T)
    return torch.stack(dynamics)


def calibrate_controller(
    model,
    tokenizer,
    *,
    model_label: str,
    model_id: str,
    cache_path: Path,
    settings: dict[str, object],
    controller_device: str,
) -> tuple[ReducedControllerArtifact, dict[str, object]]:
    """Fit or load the shared four-method controller artifact for one model."""

    fingerprint_payload = {
        "model_label": model_label,
        "model_id": model_id,
        "settings": settings,
        "calibration_version": "reduced_whitened_residual_geometry_v1",
    }
    fingerprint = configuration_hash(fingerprint_payload)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached["fingerprint"] != fingerprint:
            raise ValueError(f"Incompatible controller cache: {cache_path}")
        return ReducedControllerArtifact(**cached["artifact"]), cached["metadata"]

    seed = int(settings["seed"])
    fit_per_class = int(settings["fit_prompts_per_class"])
    calibration_count = int(settings["disturbance_prompts"])
    all_rtp, toxic, nontoxic = load_real_toxicity_prompt_pools(RTP_ID, RTP_REVISION)
    rng = random.Random(seed)
    fit_toxic = stable_sample(toxic, fit_per_class, rng)
    fit_nontoxic = stable_sample(nontoxic, fit_per_class, rng)
    fit_ids = {str(row["prompt_id"]) for row in fit_toxic + fit_nontoxic}
    calibration_pool = [row for row in all_rtp if str(row["prompt_id"]) not in fit_ids]
    calibration_records = stable_sample(calibration_pool, calibration_count, rng)

    fit_records = fit_toxic + fit_nontoxic
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
        [str(row["text"]) for row in calibration_records],
        max_length=int(settings["calibration_max_length"]),
        batch_size=int(settings["activation_batch_size"]),
    )
    means, encoders_all, decoders_all, fit_reduced = _fit_reduced_coordinates(
        fit_states,
        int(settings["state_rank"]),
        float(settings["whitening_floor"]),
    )
    calibration_reduced = _apply_coordinates(
        calibration_states,
        means,
        encoders_all,
    )
    dynamics = _fit_dynamics(fit_reduced, float(settings["ridge"]))
    predicted = torch.einsum(
        "lij,nlj->nli", dynamics, calibration_reduced[:, :-1]
    )
    residuals = calibration_reduced[:, 1:] - predicted
    disturbance = fit_disturbance_geometry(
        residuals,
        variance_threshold=float(settings["disturbance_variance"]),
    )

    toxic_reduced = fit_reduced[:fit_per_class]
    nontoxic_reduced = fit_reduced[fit_per_class:]
    contrast = nontoxic_reduced.mean(dim=0) - toxic_reduced.mean(dim=0)
    contrast_norm = torch.linalg.vector_norm(contrast, dim=1).clamp_min(1e-8)
    feature_unit_all = contrast / contrast_norm.unsqueeze(1)
    setpoints_all = float(settings["setpoint_multiplier"]) * contrast_norm

    horizon, state_dimension, _ = dynamics.shape
    identity = torch.eye(state_dimension)
    control_channels = identity.unsqueeze(0).repeat(horizon, 1, 1)
    depth_weight = 1.0 / horizon
    state_costs = (
        float(settings["q"]) * depth_weight * identity
    ).unsqueeze(0).repeat(horizon, 1, 1)
    control_costs = (float(settings["r"]) * identity).unsqueeze(0).repeat(
        horizon, 1, 1
    )
    terminal_cost = float(settings["q_final"]) * identity

    lqr_gains = solve_identity_input_lqr(
        dynamics,
        controller_device,
        state_cost=float(settings["q"]) * depth_weight,
        control_cost=float(settings["r"]),
        terminal_cost=float(settings["q_final"]),
    )
    problem = FiniteHorizonControlProblem(
        dynamics=dynamics,
        control_channels=control_channels,
        disturbance_channels=disturbance.channels,
        state_costs=state_costs,
        control_costs=control_costs,
        terminal_cost=terminal_cost,
    )
    hinf = HInfinityController.synthesize(
        problem,
        device=controller_device,
        options=HInfinityOptions(
            gamma_lower=float(settings["gamma_lower"]),
            gamma_upper=float(settings["gamma_upper"]),
            tolerance=float(settings["gamma_tolerance"]),
            max_iterations=int(settings["gamma_max_iterations"]),
            deployment_margin=float(settings["gamma_deployment_margin"]),
        ),
    ).solution()
    artifact = ReducedControllerArtifact(
        means=means[:-1],
        encoders=encoders_all[:-1],
        decoders=decoders_all[1:],
        feature_unit=feature_unit_all[:-1],
        setpoints=setpoints_all[:-1],
        control_channels=control_channels,
        lqr_gains=lqr_gains,
        hinf_gains=hinf.gains,
        hinf_feasible=hinf.feasible,
        gamma_star=hinf.gamma_star,
        hinf_diagnostics=hinf.diagnostics,
    )
    metadata = {
        "fingerprint": fingerprint,
        "model_label": model_label,
        "model_id": model_id,
        "fit_prompt_ids": sorted(fit_ids),
        "calibration_prompt_ids": [str(row["prompt_id"]) for row in calibration_records],
        "state_rank": state_dimension,
        "horizon": horizon,
        "disturbance_ranks": disturbance.retained_ranks.tolist(),
        "disturbance_explained_variance": disturbance.explained_variance.tolist(),
        "gamma_star": hinf.gamma_star,
        "robust_steerability": (
            None if hinf.gamma_star is None else 1.0 / hinf.gamma_star
        ),
        "hinf_feasible": hinf.feasible,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fingerprint": fingerprint,
            "artifact": artifact.__dict__,
            "metadata": metadata,
        },
        cache_path,
    )
    return artifact, metadata
