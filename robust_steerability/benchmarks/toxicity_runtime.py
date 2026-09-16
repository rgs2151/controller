"""Run the RealToxicityPrompts benchmark in an explicit KV-cache mode."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from robust_steerability.calibration.nominal_artifact import nominal_dynamics_signature
from robust_steerability.benchmarks.composition import load_composition
from robust_steerability.benchmarks.launcher import run_data_shards
from robust_steerability.benchmarks.layout import (
    artifact_root,
    calibration_root,
    dataset_root,
    evaluation_root,
    results_root,
)
from robust_steerability.benchmarks.specs import MODELS
from robust_steerability.benchmarks.metrics import distinct_ngrams, toxicity_frequency
from robust_steerability.datasets.toxicity import (
    load_real_toxicity_prompt_pools,
    toxicity_probabilities,
)
from robust_steerability.control import (
    FiniteHorizonControlProblem,
    HInfinityController,
    HInfinityOptions,
)
from robust_steerability.experiments.calibration import (
    calibrate_controller,
    diagnostic_root,
)
from robust_steerability.experiments.methods import ControllerArtifact, build_policy
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.judges import openai as openai_scoring
from robust_steerability.judges.exact import harmonic_mean
from robust_steerability.judges.specs import scorer_cache_path
from robust_steerability.modeling.huggingface import (
    CausalModelLoadSpec,
    load_access_token,
    load_causal_model,
    load_sequence_classifier,
)
from robust_steerability.modeling.interventions import register_generation_policy_hooks
from robust_steerability.source_methods.control import (
    SetpointCalibration,
    build_alqr_policy,
    build_spid_policy,
)
from robust_steerability.source_methods.generation import generate_batched
from robust_steerability.source_methods.id_benchmark import (
    _batch_assignments,
    generation_shard_path,
    merge_generation_shards,
    runtime_provenance,
)
from robust_steerability.source_methods.modeling import load_source_model, source_model_spec
from robust_steerability.source_methods.protocol import (
    ALQR_CALIBRATION_COUNTS,
    GENERATION,
    SOURCE_RANDOM_SEED,
    SPID_SOURCE_GRIDS,
    paper_alqr_setting,
)


REPO = Path(__file__).resolve().parents[2]
MODEL_KEY = "gemma2b"
CALIBRATION_ID = "selected"
MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
MODEL_LABEL = "Gemma-2-2B"
CURRENT_USE_CACHE = False
CACHE = evaluation_root("toxicity", MODEL_KEY, use_cache=CURRENT_USE_CACHE)
SHARED = artifact_root("toxicity", MODEL_KEY)
HINF_CALIBRATION = calibration_root("toxicity", MODEL_KEY, "h_infinity")
SPID_CALIBRATION = calibration_root("toxicity", MODEL_KEY, "spid")
DATA_PATH = dataset_root("toxicity", MODEL_KEY) / "toxicity_rtp.json"
RTP_ID = "allenai/real-toxicity-prompts"
RTP_REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
TOXICITY_CLASSIFIER = "s-nlp/roberta_toxicity_classifier"
TOXICITY_CLASSIFIER_REVISION = "048c25bb1e199b98802784f96325f4840f22145d"
PERPLEXITY_MODEL = "mistralai/Mistral-7B-v0.1"
PERPLEXITY_MODEL_REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"

COMPOSITION = load_composition("toxicity")
METHODS = COMPOSITION.available_methods
DISTRIBUTIONS = ("rtp",)
API_SCORERS = (
    "axbench_concept_relevance",
    "axbench_instruction_relevance",
    "axbench_fluency",
)
SCORERS = (
    "toxicity_classifier",
    "distinct_2",
    "perplexity",
    *API_SCORERS,
    "axbench_overall",
)
TOXICITY_CONCEPT = "non-toxic, respectful, and non-abusive language"
EVALUATION_REPETITIONS = 5
EVALUATION_SAMPLES = 1000
CALIBRATION_REPETITIONS = 1
CALIBRATION_SAMPLES = 50
DISTURBANCE_SAMPLES = 200
GENERATION_BATCH_SIZE = 8
TOXICITY_BATCH_SIZE = 16
PERPLEXITY_BATCH_SIZE = 10
PERPLEXITY_MAX_LENGTH = 128
SEED_STRIDE = 100_000
Q_OVER_R = (0.01, 0.1, 1.0, 10.0)
Q_FINAL_OVER_R = (0.01, 0.1, 10**-0.5)
FIXED_R = 1.0
FIXED_SETPOINT_MULTIPLIER = 3.5


def configure_model(
    model_key: str,
    calibration_id: str = "selected",
    generation_batch_size: int | None = None,
    *,
    use_cache: bool = False,
) -> None:
    """Bind this worker process to exactly one model-owned cache tree."""

    if model_key not in COMPOSITION.models:
        raise ValueError(f"Unknown model {model_key!r}")
    spec = MODELS[model_key]
    global MODEL_KEY, MODEL_ID, MODEL_REVISION, MODEL_LABEL, CALIBRATION_ID
    global CACHE, SHARED, HINF_CALIBRATION, SPID_CALIBRATION, DATA_PATH
    global CURRENT_USE_CACHE
    global FIXED_SETPOINT_MULTIPLIER, GENERATION_BATCH_SIZE
    MODEL_KEY = spec.key
    if not calibration_id or "/" in calibration_id:
        raise ValueError("calibration_id must be a simple name")
    if generation_batch_size is not None and generation_batch_size < 1:
        raise ValueError("generation_batch_size must be positive")
    CALIBRATION_ID = calibration_id
    MODEL_ID = spec.model_id
    MODEL_REVISION = spec.revision
    MODEL_LABEL = spec.label
    CURRENT_USE_CACHE = use_cache
    CACHE = evaluation_root("toxicity", model_key, use_cache=use_cache)
    SHARED = artifact_root("toxicity", model_key)
    HINF_CALIBRATION = calibration_root(
        "toxicity", model_key, "h_infinity", calibration_id
    )
    SPID_CALIBRATION = calibration_root(
        "toxicity", model_key, "spid", calibration_id
    )
    DATA_PATH = dataset_root("toxicity", model_key) / "toxicity_rtp.json"
    FIXED_SETPOINT_MULTIPLIER = paper_alqr_setting(
        "toxicity", MODEL_ID
    ).multiplier
    GENERATION_BATCH_SIZE = generation_batch_size or 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _write_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sample(records: list[dict], count: int, seed: int) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"Requested {count} records from a pool of {len(records)}")
    return [records[index] for index in random.Random(seed).sample(range(len(records)), count)]


def _source_id(record: dict) -> str:
    return str(record.get("source_prompt_id", record.get("question_id", record["prompt_id"])))


def prepare() -> None:
    """Freeze disjoint RTP fit, tuning, and evaluation records."""

    if DATA_PATH.exists():
        return
    shared_data_path = SHARED / "data.json"
    if not shared_data_path.exists():
        raise FileNotFoundError(
            "Run the toxicity artifact stage for this model first"
        )
    shared_data = json.loads(shared_data_path.read_text())
    all_rtp, _toxic, _nontoxic = load_real_toxicity_prompt_pools(RTP_ID, RTP_REVISION)

    semantic_records = [
        row
        for split in ("undesired", "desired", "jacobian")
        for row in shared_data["calibration"][split]
    ]
    semantic_ids = {_source_id(row) for row in semantic_records}
    disturbance_pool = [row for row in all_rtp if _source_id(row) not in semantic_ids]
    disturbance = _sample(
        disturbance_pool, DISTURBANCE_SAMPLES, SOURCE_RANDOM_SEED + 3
    )
    reserved = semantic_ids | {_source_id(row) for row in disturbance}
    tuning_pool = [row for row in all_rtp if _source_id(row) not in reserved]
    tuning = _sample(
        tuning_pool,
        CALIBRATION_REPETITIONS * CALIBRATION_SAMPLES,
        SOURCE_RANDOM_SEED + 4,
    )
    tuning_repetitions = {
        str(repetition): tuning[
            repetition * CALIBRATION_SAMPLES : (repetition + 1) * CALIBRATION_SAMPLES
        ]
        for repetition in range(CALIBRATION_REPETITIONS)
    }
    reserved |= {_source_id(row) for row in tuning}
    final_rtp_pool = [row for row in all_rtp if _source_id(row) not in reserved]

    payload = {
        "schema_version": 2,
        "behavior": "toxicity",
        "model": [MODEL_ID, MODEL_REVISION],
        "seed": SOURCE_RANDOM_SEED,
        "datasets": {"rtp": [RTP_ID, RTP_REVISION, "train"]},
        "split_policy": (
            "semantic fit, H-infinity disturbance fit, hyperparameter tuning, and final "
            "RTP evaluation use disjoint prompt IDs; final repetitions may overlap each other"
        ),
        "calibration": {
            "negative": shared_data["calibration"]["undesired"],
            "positive": shared_data["calibration"]["desired"],
            "jacobian": shared_data["calibration"]["jacobian"],
            "disturbance": disturbance,
            "dataset": {
                "id": RTP_ID,
                "revision": RTP_REVISION,
                "shared_artifact_data": str(shared_data_path.relative_to(REPO)),
            },
        },
        "hyperparameter_evaluation": {"rtp": tuning_repetitions},
        "evaluation": {
            "rtp": {
                str(repetition): _sample(
                    final_rtp_pool,
                    EVALUATION_SAMPLES,
                    SOURCE_RANDOM_SEED + repetition * SEED_STRIDE,
                )
                for repetition in range(EVALUATION_REPETITIONS)
            },
        },
    }
    _write_json(DATA_PATH, payload)

def _hinf_settings() -> dict[str, object]:
    counts = ALQR_CALIBRATION_COUNTS["toxicity"]
    model = MODELS[MODEL_KEY]
    model_loading = asdict(source_model_spec("alqr", "toxicity", MODEL_ID, MODEL_REVISION))
    return {
        "behavior": "toxicity_mitigation",
        "seed": SOURCE_RANDOM_SEED,
        "fit_prompts_per_class": counts.undesired,
        "disturbance_prompts": DISTURBANCE_SAMPLES,
        "calibration_max_length": 128,
        "activation_batch_size": model.activation_batch_size,
        "jacobian_prompts": counts.jacobian,
        "jacobian_max_length": counts.jacobian_max_length,
        "jacobian_vjp_chunk_size": model.jacobian_vjp_chunk_size,
        "state_rank": 8,
        "numerical_floor": 1e-4,
        "alqr_setpoint_multiplier": FIXED_SETPOINT_MULTIPLIER,
        "spid_setpoint_multiplier": 1.0,
        "hinf_setpoint_multiplier": FIXED_SETPOINT_MULTIPLIER,
        "q": 0.1,
        "r": FIXED_R,
        "q_final": 0.1,
        "alqr_q": 0.1,
        "alqr_r": 1.0,
        "alqr_q_final": 0.1,
        "kp": 0.7,
        "ki": 0.01,
        "kd": 0.1,
        "gamma_lower": 0.0,
        "gamma_upper": 100.0,
        "gamma_tolerance": 1e-5,
        "gamma_max_iterations": 100,
        "gamma_deployment_margin": 0.01,
        "model_loading": model_loading,
    }


def calibrate_hinf_base(device: str) -> None:
    """Fit D and the rank-8 coordinates once while reusing the A-LQR A matrix."""

    prepare()
    nominal_path = SHARED / "dynamics.pt"
    if not nominal_path.exists():
        raise FileNotFoundError(f"Missing shared A matrix: {nominal_path}")
    data = json.loads(DATA_PATH.read_text())
    model, tokenizer = load_source_model(
        "alqr", "toxicity", MODEL_ID, MODEL_REVISION, device, load_access_token(REPO)
    )
    calibrate_controller(
        model,
        tokenizer,
        model_label=MODEL_LABEL,
        model_id=MODEL_ID,
        cache_path=HINF_CALIBRATION / "base/controller.pt",
        nominal_dynamics_path=nominal_path,
        calibration_data=data["calibration"],
        settings=_hinf_settings(),
        controller_device=device,
    )


def _grid() -> list[dict[str, float | str]]:
    return [
        {
            "grid_id": f"q_{q_index:02d}_qf_{qf_index:02d}",
            "q": float(q_ratio * FIXED_R),
            "r": FIXED_R,
            "q_final": float(qf_ratio * FIXED_R),
            "q_over_r": float(q_ratio),
            "q_final_over_r": float(qf_ratio),
            "lambda": FIXED_SETPOINT_MULTIPLIER,
        }
        for q_index, q_ratio in enumerate(Q_OVER_R)
        for qf_index, qf_ratio in enumerate(Q_FINAL_OVER_R)
    ]


def _base_payload() -> tuple[ControllerArtifact, dict, HInfinityOptions]:
    base_path = HINF_CALIBRATION / "base/controller.pt"
    input_path = diagnostic_root(base_path) / "input.pt"
    if not base_path.exists() or not input_path.exists():
        raise FileNotFoundError("H-infinity base calibration is incomplete")
    payload = torch.load(base_path, map_location="cpu", weights_only=True, mmap=True)
    metadata = payload.get("metadata", {})
    nominal_path = SHARED / "dynamics.pt"
    if metadata.get("nominal_dynamics") != nominal_dynamics_signature(nominal_path):
        raise ValueError(
            "Frozen H-infinity controller was not synthesized from the current "
            "shared A-LQR dynamics artifact"
        )
    bundle = torch.load(input_path, map_location="cpu", weights_only=True, mmap=True)
    return (
        ControllerArtifact(**payload["artifact"]),
        bundle,
        HInfinityOptions(**bundle["options"]),
    )


def synthesize_hinf_grid(device: str) -> None:
    """Synthesize the frozen 8-by-4 Q/R--Qf/R calibration grid."""

    base, bundle, options = _base_payload()
    source_problem = FiniteHorizonControlProblem(**bundle["problem"])
    source_settings = bundle["calibration"]["settings"]
    for configuration in _grid():
        destination = HINF_CALIBRATION / "grid/controllers" / f"{configuration['grid_id']}.pt"
        identity = {
            "schema_version": 1,
            "model": [MODEL_ID, MODEL_REVISION],
            "task": "toxicity",
            "kv_cache": False,
            "configuration": configuration,
        }
        if destination.exists():
            continue
        problem = FiniteHorizonControlProblem(
            dynamics=source_problem.dynamics,
            control_channels=source_problem.control_channels,
            disturbance_channels=source_problem.disturbance_channels,
            state_costs=source_problem.state_costs
            * (float(configuration["q"]) / float(source_settings["q"])),
            control_costs=source_problem.control_costs
            * (float(configuration["r"]) / float(source_settings["r"])),
            terminal_cost=source_problem.terminal_cost
            * (float(configuration["q_final"]) / float(source_settings["q_final"])),
            metadata={"configuration": configuration, "shared_A_D_and_coordinates": True},
        )
        started = time.perf_counter()
        solution = HInfinityController.synthesize(
            problem, device=device, options=options
        ).solution()
        if not solution.feasible or solution.gamma_star is None:
            raise ValueError(f"Infeasible H-infinity grid point: {configuration}")
        _write_torch(
            destination,
            {
                "identity": identity,
                "gains": solution.gains,
                "feasible": solution.feasible,
                "gamma_star": solution.gamma_star,
                "diagnostics": solution.diagnostics,
                "elapsed_seconds": time.perf_counter() - started,
                "runtime": runtime_provenance(device),
            },
        )


def _load_setpoint() -> SetpointCalibration:
    payload = torch.load(SHARED / "setpoint.pt", map_location="cpu", weights_only=True)
    return SetpointCalibration(
        contrast=payload["contrast"], feature_norm=payload["feature_norm"]
    )


def _alqr_policy(device: str):
    dynamics = torch.load(
        SHARED / "dynamics.pt", map_location="cpu", weights_only=True, mmap=True
    )["dynamics"]
    setting = paper_alqr_setting("toxicity", MODEL_ID)
    return build_alqr_policy(
        dynamics,
        _load_setpoint(),
        multiplier=setting.multiplier,
        q=setting.q,
        r=setting.r,
        q_final=setting.q_final,
        device=device,
    )


def _spid_policy(multiplier: float):
    grid = SPID_SOURCE_GRIDS["toxicity"][MODEL_KEY]
    return build_spid_policy(
        _load_setpoint(),
        multiplier=multiplier,
        kp=grid.kp,
        ki=grid.ki,
        kd=grid.kd,
    )


def _hinf_grid_policy(grid_id: str):
    base, _bundle, _options = _base_payload()
    variant = torch.load(
        HINF_CALIBRATION / "grid/controllers" / f"{grid_id}.pt",
        map_location="cpu",
        weights_only=True,
    )
    artifact = replace(
        base,
        hinf_gains=variant["gains"],
        hinf_feasible=variant["feasible"],
        gamma_star=variant["gamma_star"],
        hinf_diagnostics=variant["diagnostics"],
    )
    return build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)


def _selected_policy(method: str, device: str):
    if method == "original":
        return None
    if method == "alqr":
        return _alqr_policy(device)
    if method == "spid":
        selection = json.loads((SPID_CALIBRATION / "selection.json").read_text())
        return _spid_policy(float(selection["parameters"]["lambda"]))
    _base_payload()
    payload = torch.load(
        HINF_CALIBRATION / "controller.pt",
        map_location="cpu",
        weights_only=True,
    )
    artifact = ControllerArtifact(**payload["artifact"])
    return build_policy("hinf", artifact, kp=0.0, ki=0.0, kd=0.0)


def _candidate_specs() -> list[dict]:
    return [
        {
            "candidate_id": str(configuration["grid_id"]),
            "method": "h_infinity",
            "parameters": configuration,
        }
        for configuration in _grid()
    ]


def _candidate_policy(candidate: dict, device: str):
    return _hinf_grid_policy(str(candidate["candidate_id"]))


def _generate_records(
    model,
    tokenizer,
    policy,
    records: list[dict],
    seed: int,
    *,
    use_cache: bool,
) -> list[dict]:
    register = (
        None
        if policy is None
        else lambda: register_generation_policy_hooks(model, policy)
    )
    completions = generate_batched(
        model,
        tokenizer,
        [str(row["text"]) for row in records],
        behavior="toxicity",
        batch_size=GENERATION_BATCH_SIZE,
        seed=seed,
        use_cache=use_cache,
        register_hooks=register,
        reset=None if policy is None else policy.reset,
    )
    return [
        {
            "prompt_id": str(row["prompt_id"]),
            "text": str(row["text"]),
            "completion": completion,
            "concept": TOXICITY_CONCEPT,
        }
        for row, completion in zip(records, completions, strict=True)
    ]


def generate_calibration_worker(device: str, worker_index: int, worker_count: int) -> None:
    """Generate assigned H-infinity development configurations."""

    prepare()
    candidates = _candidate_specs()[worker_index::worker_count]
    model, tokenizer = load_source_model(
        "alqr", "toxicity", MODEL_ID, MODEL_REVISION, device, load_access_token(REPO)
    )
    data = json.loads(DATA_PATH.read_text())
    for candidate in candidates:
        destination = HINF_CALIBRATION / "grid/generations" / f"{candidate['candidate_id']}.json"
        identity = {
            "schema_version": 1,
            "candidate": candidate,
            "kv_cache": False,
            "generation": GENERATION["toxicity"],
            "batch_size": GENERATION_BATCH_SIZE,
        }
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get("status") == "complete":
                continue
        else:
            saved = {
                "identity": identity,
                "status": "partial",
                "repetitions": [],
                "attempts": [],
            }
        policy = _candidate_policy(candidate, device)
        started = time.perf_counter()
        attempt = {
            "started_at_utc": _utc_now(),
            "status": "running",
            "runtime": runtime_provenance(device),
        }
        saved["attempts"].append(attempt)
        _write_json(destination, saved)
        for repetition in range(len(saved["repetitions"]), CALIBRATION_REPETITIONS):
            rows = _generate_records(
                model,
                tokenizer,
                policy,
                data["hyperparameter_evaluation"]["rtp"][str(repetition)],
                SOURCE_RANDOM_SEED + repetition * SEED_STRIDE,
                use_cache=False,
            )
            saved["repetitions"].append({"repetition": repetition, "rows": rows})
            _write_json(destination, saved)
        attempt["finished_at_utc"] = _utc_now()
        attempt["elapsed_seconds"] = time.perf_counter() - started
        attempt["status"] = "complete"
        saved["status"] = "complete"
        _write_json(destination, saved)


def _load_perplexity_model(device: str, token: str):
    return load_causal_model(
        CausalModelLoadSpec(
            model_id=PERPLEXITY_MODEL,
            revision=PERPLEXITY_MODEL_REVISION,
            quantized=True,
            dtype="float32",
            attention_implementation=None,
            quantization_compute_dtype="float16",
        ),
        device,
        token,
    )


def _perplexities(model, tokenizer, texts: list[str], device: str) -> list[float]:
    values = []
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = "right"
    for start in range(0, len(texts), PERPLEXITY_BATCH_SIZE):
        encoded = tokenizer(
            texts[start : start + PERPLEXITY_BATCH_SIZE],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=PERPLEXITY_MAX_LENGTH,
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits.float()
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            encoded["input_ids"][:, 1:].reshape(-1),
            reduction="none",
        ).reshape(logits.shape[0], -1)
        mask = encoded["attention_mask"][:, 1:]
        values.extend(
            torch.exp((losses * mask).sum(-1) / mask.sum(-1).clamp_min(1))
            .detach()
            .cpu()
            .tolist()
        )
    tokenizer.padding_side = original_padding
    return [float(value) for value in values]


def _write_axbench_overall(root: Path, generation_path: Path) -> Path:
    component_rows = []
    for scorer in API_SCORERS:
        payload = json.loads(scorer_cache_path(root, generation_path, scorer).read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete AXBench score: {scorer}")
        component_rows.append(payload["rows"])
    lengths = {len(rows) for rows in component_rows}
    if len(lengths) != 1:
        raise ValueError("AXBench component scorers returned different row counts")
    rows = [
        {
            "prompt_id": component_rows[0][index]["prompt_id"],
            "score": harmonic_mean(
                [float(component[index]["score"]) for component in component_rows]
            ),
        }
        for index in range(len(component_rows[0]))
    ]
    destination = scorer_cache_path(root, generation_path, "axbench_overall")
    _write_json(destination, {"status": "complete", "rows": rows})
    return destination


def score_calibration_grid(*, api_concurrency: int, api_batch_size: int) -> None:
    """Score every H-infinity candidate with the three AXBench judges."""

    root = HINF_CALIBRATION / "grid"
    generation_paths = [
        root / "generations" / f"{candidate['candidate_id']}.json"
        for candidate in _candidate_specs()
    ]
    if any(not path.exists() for path in generation_paths):
        raise FileNotFoundError("H-infinity calibration generations are incomplete")
    openai_scoring.score_generations(
        generation_paths,
        root,
        list(API_SCORERS),
        concurrency=api_concurrency,
        batch_size=api_batch_size,
    )
    for generation_path in generation_paths:
        _write_axbench_overall(root, generation_path)


def _candidate_summary(candidate: dict) -> dict:
    generation_path = HINF_CALIBRATION / "grid/generations" / f"{candidate['candidate_id']}.json"
    generation = json.loads(generation_path.read_text())
    if generation.get("status") != "complete":
        raise ValueError(f"Incomplete calibration candidate: {candidate['candidate_id']}")
    root = HINF_CALIBRATION / "grid"
    means = {}
    for scorer in (*API_SCORERS, "axbench_overall"):
        payload = json.loads(scorer_cache_path(root, generation_path, scorer).read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete calibration score: {scorer}")
        means[scorer] = float(np.mean([float(row["score"]) for row in payload["rows"]]))
    return {
        **candidate,
        **means,
    }


def select_calibration() -> None:
    """Freeze the H-infinity configuration with highest AXBench overall score."""

    summaries = [_candidate_summary(candidate) for candidate in _candidate_specs()]
    hinf = sorted(
        summaries,
        key=lambda row: (
            -row["axbench_overall"],
            -row["axbench_concept_relevance"],
            -row["axbench_instruction_relevance"],
            -row["axbench_fluency"],
            row["parameters"]["q_over_r"],
            row["parameters"]["q_final_over_r"],
        ),
    )[0]
    base, _bundle, _options = _base_payload()
    variant_path = HINF_CALIBRATION / "grid/controllers" / f"{hinf['candidate_id']}.pt"
    variant = torch.load(variant_path, map_location="cpu", weights_only=True)
    selected_artifact = replace(
        base,
        hinf_gains=variant["gains"],
        hinf_feasible=variant["feasible"],
        gamma_star=variant["gamma_star"],
        hinf_diagnostics=variant["diagnostics"],
    )
    controller_path = HINF_CALIBRATION / "controller.pt"
    _write_torch(
        controller_path,
        {
            "identity": {
                "schema_version": 1,
                "model": [MODEL_ID, MODEL_REVISION],
                "task": "toxicity",
                "kv_cache": False,
                "selection_candidate": hinf["candidate_id"],
                "parameters": hinf["parameters"],
            },
            "artifact": selected_artifact.__dict__,
        },
    )
    _write_json(
        HINF_CALIBRATION / "selection.json",
        {
            "schema_version": 1,
            "model": [MODEL_ID, MODEL_REVISION],
            "benchmark": "toxicity",
            "method": "h_infinity",
            "calibration_id": CALIBRATION_ID,
            "dataset": [RTP_ID, RTP_REVISION],
            "kv_cache": False,
            "calibration": {
                "repetitions": CALIBRATION_REPETITIONS,
                "samples_per_repetition": CALIBRATION_SAMPLES,
                "q_over_r": list(Q_OVER_R),
                "q_final_over_r": list(Q_FINAL_OVER_R),
                "fixed_r": FIXED_R,
            },
            "selection_rule": "maximize mean AXBench overall steering",
            "tie_breakers": [
                "higher concept relevance",
                "higher instruction relevance",
                "higher fluency",
                "smaller Q/R",
                "smaller Qf/R",
            ],
            "selected": hinf,
            "grid": summaries,
        },
    )


def _generation_path(distribution: str, method: str) -> Path:
    return CACHE / "generations" / distribution / method / "final.json"


def generate_final(
    method: str,
    device: str,
    distributions: tuple[str, ...] = DISTRIBUTIONS,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> None:
    """Generate one data shard for a frozen method on each distribution."""

    if method not in METHODS:
        raise ValueError(f"Unknown final method: {method}")
    if shard_index is None or shard_count is None:
        raise ValueError("Final toxicity generation requires shard index and count")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Final toxicity generation requires a valid shard set")
    prepare()
    required_selection = {
        "alqr": calibration_root(
            "toxicity", MODEL_KEY, "alqr", CALIBRATION_ID
        ) / "selection.json",
        "spid": SPID_CALIBRATION / "selection.json",
        "h_infinity": HINF_CALIBRATION / "selection.json",
    }.get(method)
    if required_selection is not None and not required_selection.exists():
        raise FileNotFoundError(f"Run {method} calibration selection first")
    data = json.loads(DATA_PATH.read_text())
    model, tokenizer = load_source_model(
        "alqr" if method in {"original", "spid", "alqr", "h_infinity"} else method,
        "toxicity",
        MODEL_ID,
        MODEL_REVISION,
        device,
        load_access_token(REPO),
    )
    parameters = _method_parameters(method)
    policy = _selected_policy(method, device)
    for distribution in distributions:
        destination = generation_shard_path(
            CACHE, distribution, method, shard_index, shard_count
        )
        common_identity = {
            "schema_version": 1,
            "model": [MODEL_ID, MODEL_REVISION],
            "method": method,
            "parameters": parameters,
            "distribution": distribution,
            "kv_cache": CURRENT_USE_CACHE,
            "generation": GENERATION["toxicity"],
            "batch_size": GENERATION_BATCH_SIZE,
        }
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved.get("status") == "complete":
                continue
        else:
            saved = {
                "identity": {
                    "base": common_identity,
                    "shard": {"index": shard_index, "count": shard_count},
                },
                "status": "partial",
                "batches": [],
                "attempts": [],
            }
        started = time.perf_counter()
        attempt = {
            "started_at_utc": _utc_now(),
            "status": "running",
            "runtime": runtime_provenance(device),
        }
        saved["attempts"].append(attempt)
        _write_json(destination, saved)
        assignments = _batch_assignments(data, distribution, GENERATION_BATCH_SIZE)
        assigned = [
            assignment
            for index, assignment in enumerate(assignments)
            if index % shard_count == shard_index
        ]
        completed = [
            (int(row["repetition"]), int(row["start"]))
            for row in saved["batches"]
        ]
        expected = [(repetition, start) for repetition, start, _records in assigned]
        if completed != expected[: len(completed)]:
            raise ValueError(f"Toxicity shard batches are not a valid prefix: {destination}")
        for repetition, start, records in assigned[len(completed) :]:
            batch_started = time.perf_counter()
            started_at = _utc_now()
            rows = _generate_records(
                model,
                tokenizer,
                policy,
                records,
                SOURCE_RANDOM_SEED + repetition * SEED_STRIDE + start,
                use_cache=CURRENT_USE_CACHE,
            )
            saved["batches"].append(
                {
                    "repetition": repetition,
                    "start": start,
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "elapsed_seconds": time.perf_counter() - batch_started,
                    "generation_seed": SOURCE_RANDOM_SEED + repetition * SEED_STRIDE + start,
                    "rows": rows,
                }
            )
            _write_json(destination, saved)
        attempt["finished_at_utc"] = _utc_now()
        attempt["elapsed_seconds"] = time.perf_counter() - started
        attempt["status"] = "complete"
        saved["status"] = "complete"
        _write_json(destination, saved)


def _method_parameters(method: str) -> dict:
    if method == "original":
        return {}
    if method == "alqr":
        setting = paper_alqr_setting("toxicity", MODEL_ID)
        return {
            "lambda": setting.multiplier,
            "q": setting.q,
            "r": setting.r,
            "q_final": setting.q_final,
        }
    if method == "spid":
        path = SPID_CALIBRATION / "selection.json"
    else:
        path = HINF_CALIBRATION / "selection.json"
    selection = json.loads(path.read_text())
    if (
        selection.get("schema_version") != 1
        or selection.get("model") != [MODEL_ID, MODEL_REVISION]
        or selection.get("benchmark") != "toxicity"
        or selection.get("method") != method
        or selection.get("calibration_id") != CALIBRATION_ID
        or selection.get("kv_cache") is not False
    ):
        raise ValueError(f"Invalid selected {method} calibration: {path}")
    return (
        selection["parameters"]
        if method == "spid"
        else selection["selected"]["parameters"]
    )


def _score_path(scorer: str, distribution: str, method: str) -> Path:
    return CACHE / "scores" / scorer / distribution / method / "final.json"


def score_final(
    method: str,
    device: str,
    distributions: tuple[str, ...] = DISTRIBUTIONS,
    scorers: tuple[str, ...] = SCORERS,
    *,
    api_concurrency: int = openai_scoring.DEFAULT_CONCURRENCY,
    api_batch_size: int = openai_scoring.DEFAULT_BATCH_SIZE,
) -> None:
    """Run only the selected benchmark scorers against cached generations."""

    unknown = set(scorers) - set(SCORERS)
    if unknown:
        raise ValueError(f"Unknown toxicity scorers: {sorted(unknown)}")
    token = (
        load_access_token(REPO)
        if {"toxicity_classifier", "perplexity"} & set(scorers)
        else ""
    )
    pending_toxicity = [
        value for value in distributions
        if "toxicity_classifier" in scorers
        and not _score_path("toxicity_classifier", value, method).exists()
    ]
    if pending_toxicity:
        classifier, classifier_tokenizer = load_sequence_classifier(
            TOXICITY_CLASSIFIER,
            TOXICITY_CLASSIFIER_REVISION,
            device,
            token,
        )
        for distribution in pending_toxicity:
            generation = json.loads(
                _generation_path(distribution, method).read_text()
            )
            flat = [
                row for repetition in generation["repetitions"]
                for row in repetition["rows"]
            ]
            probabilities = toxicity_probabilities(
                [str(row["completion"]) for row in flat],
                classifier,
                classifier_tokenizer,
                device,
                batch_size=TOXICITY_BATCH_SIZE,
                max_length=512,
            )
            _write_json(
                _score_path("toxicity_classifier", distribution, method),
                {
                    "status": "complete",
                    "scorer": "toxicity_classifier",
                    "model": [TOXICITY_CLASSIFIER, TOXICITY_CLASSIFIER_REVISION],
                    "values": [float(value) for value in probabilities],
                },
            )
        del classifier, classifier_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    pending_perplexity = [
        value for value in distributions
        if "perplexity" in scorers
        and not _score_path("perplexity", value, method).exists()
    ]
    if pending_perplexity:
        perplexity_model, perplexity_tokenizer = _load_perplexity_model(device, token)
        for distribution in pending_perplexity:
            generation = json.loads(
                _generation_path(distribution, method).read_text()
            )
            flat = [
                row for repetition in generation["repetitions"]
                for row in repetition["rows"]
            ]
            _write_json(
                _score_path("perplexity", distribution, method),
                {
                    "status": "complete",
                    "scorer": "perplexity",
                    "model": [PERPLEXITY_MODEL, PERPLEXITY_MODEL_REVISION],
                    "values": _perplexities(
                        perplexity_model,
                        perplexity_tokenizer,
                        [str(row["text"]) + str(row["completion"]) for row in flat],
                        device,
                    ),
                },
            )
        del perplexity_model, perplexity_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    if "distinct_2" in scorers:
        for distribution in distributions:
            destination = _score_path("distinct_2", distribution, method)
            if destination.exists():
                continue
            generation = json.loads(
                _generation_path(distribution, method).read_text()
            )
            _write_json(
                destination,
                {
                    "status": "complete",
                    "scorer": "distinct_2",
                    "definition": "unique generated bigrams divided by generated bigrams",
                    "values": [
                        distinct_ngrams(
                            [str(row["completion"]) for row in repetition["rows"]],
                            2,
                        )
                        for repetition in generation["repetitions"]
                    ],
                },
            )

    requested_api = set(scorers) & set(API_SCORERS)
    if "axbench_overall" in scorers:
        requested_api.update(API_SCORERS)
    if requested_api:
        generation_paths = [
            _generation_path(distribution, method) for distribution in distributions
        ]
        openai_scoring.score_generations(
            generation_paths,
            CACHE,
            sorted(requested_api),
            concurrency=api_concurrency,
            batch_size=api_batch_size,
        )
        if "axbench_overall" in scorers:
            for generation_path in generation_paths:
                _write_axbench_overall(CACHE, generation_path)


def _mean_se(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "standard_error": float(array.std(ddof=1) / math.sqrt(len(array))),
    }


def summarize_final(
    method: str,
    distribution: str,
    scorers: tuple[str, ...] = SCORERS,
) -> dict:
    generation_path = _generation_path(distribution, method)
    generation = json.loads(generation_path.read_text())
    scores = {
        scorer: json.loads(_score_path(scorer, distribution, method).read_text())
        for scorer in scorers
    }
    per_repetition = []
    offset = 0
    for repetition in generation["repetitions"]:
        count = len(repetition["rows"])
        row = {"repetition": repetition["repetition"]}
        if "toxicity_classifier" in scorers:
            probabilities = scores["toxicity_classifier"]["values"][offset : offset + count]
            row["toxicity"], _se = toxicity_frequency(probabilities)
        if "distinct_2" in scorers:
            row["dist_2"] = scores["distinct_2"]["values"][repetition["repetition"]]
        if "perplexity" in scorers:
            row["perplexity"] = float(
                np.mean(scores["perplexity"]["values"][offset : offset + count])
            )
        for scorer in (*API_SCORERS, "axbench_overall"):
            if scorer in scorers:
                row[scorer] = float(
                    np.mean(
                        [
                            float(value["score"])
                            for value in scores[scorer]["rows"][offset : offset + count]
                        ]
                    )
                )
        per_repetition.append(row)
        offset += count
    metric_names = {
        "toxicity_classifier": "toxicity",
        "distinct_2": "dist_2",
        "perplexity": "perplexity",
        "axbench_concept_relevance": "axbench_concept_relevance",
        "axbench_instruction_relevance": "axbench_instruction_relevance",
        "axbench_fluency": "axbench_fluency",
        "axbench_overall": "axbench_overall",
    }
    metrics = {
        metric_names[scorer]: _mean_se(
            [row[metric_names[scorer]] for row in per_repetition]
        )
        for scorer in scorers
    }
    result = {
        "identity": {
            "model": [MODEL_ID, MODEL_REVISION],
            "method": method,
            "distribution": distribution,
            "kv_cache": CURRENT_USE_CACHE,
            "scorers": list(scorers),
        },
        "evaluation_samples_per_repetition": EVALUATION_SAMPLES,
        "evaluation_repetitions": EVALUATION_REPETITIONS,
        "per_repetition": per_repetition,
        "metrics": metrics,
    }
    destination = CACHE / "results" / distribution / f"{method}.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        result["metrics"] = {**existing.get("metrics", {}), **result["metrics"]}
        existing_rows = {
            int(row["repetition"]): row for row in existing.get("per_repetition", [])
        }
        for row in result["per_repetition"]:
            existing_rows.setdefault(int(row["repetition"]), {}).update(row)
        result["per_repetition"] = [existing_rows[index] for index in sorted(existing_rows)]
        result["identity"]["scorers"] = sorted(
            set(existing.get("identity", {}).get("scorers", [])) | set(scorers)
        )
    _write_json(destination, result)
    _write_json(
        results_root("toxicity", use_cache=CURRENT_USE_CACHE)
        / MODEL_KEY
        / distribution
        / f"{method}.json",
        result,
    )
    return result


def _launch_workers(
    stage: str,
    devices: list[str],
    log_root: Path,
    *,
    candidates: bool,
    methods: tuple[str, ...] = METHODS,
    distributions: tuple[str, ...] = DISTRIBUTIONS,
    scorers: tuple[str, ...] = SCORERS,
    api_concurrency: int = openai_scoring.DEFAULT_CONCURRENCY,
    api_batch_size: int = openai_scoring.DEFAULT_BATCH_SIZE,
) -> None:
    jobs = []
    if candidates:
        jobs = [
            [
                "--stage",
                stage,
                "--model",
                MODEL_KEY,
                "--device",
                device,
                "--worker-index",
                str(index),
                "--worker-count",
                str(len(devices)),
                "--calibration-id",
                CALIBRATION_ID,
                "--generation-batch-size",
                str(GENERATION_BATCH_SIZE),
                "--kv-cache", "off",
            ]
            for index, device in enumerate(devices)
        ]
    else:
        jobs = [
            [
                "--stage", stage, "--model", MODEL_KEY,
                "--device", devices[index % len(devices)], "--method", method,
                "--distributions", ",".join(distributions),
                "--calibration-id", CALIBRATION_ID,
                "--generation-batch-size", str(GENERATION_BATCH_SIZE),
                "--kv-cache", "on" if CURRENT_USE_CACHE else "off",
                "--scorers", ",".join(scorers),
                "--api-concurrency", str(api_concurrency),
                "--api-batch-size", str(api_batch_size),
            ]
            for index, method in enumerate(methods)
        ]
    log_root.mkdir(parents=True, exist_ok=True)
    active: list[tuple[subprocess.Popen, object, Path]] = []
    for job_index, arguments in enumerate(jobs):
        if len(active) == len(devices):
            process, handle, log_path = active.pop(0)
            if process.wait() != 0:
                handle.close()
                raise RuntimeError(f"Worker failed; see {log_path}")
            handle.close()
        log_path = log_root / f"{stage}_{job_index:02d}.log"
        handle = log_path.open("a")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *arguments],
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        active.append((process, handle, log_path))
    for process, handle, log_path in active:
        if process.wait() != 0:
            handle.close()
            raise RuntimeError(f"Worker failed; see {log_path}")
        handle.close()


def calibrate(
    devices: list[str],
    *,
    log_root: Path | None = None,
    api_concurrency: int = openai_scoring.DEFAULT_CONCURRENCY,
    api_batch_size: int = openai_scoring.DEFAULT_BATCH_SIZE,
) -> None:
    prepare()
    hinf_selection_path = HINF_CALIBRATION / "selection.json"
    controller_path = HINF_CALIBRATION / "controller.pt"
    if hinf_selection_path.exists() and controller_path.exists():
        return
    calibrate_hinf_base(devices[0])
    synthesize_hinf_grid(devices[0])
    worker_logs = log_root or CACHE / "logs"
    _launch_workers(
        "generate-calibration-worker", devices, worker_logs, candidates=True
    )
    score_calibration_grid(
        api_concurrency=api_concurrency,
        api_batch_size=api_batch_size,
    )
    select_calibration()


def evaluate(
    devices: list[str],
    *,
    methods: list[str] | tuple[str, ...] = METHODS,
    distributions: list[str] | tuple[str, ...] = DISTRIBUTIONS,
    log_root: Path | None = None,
) -> None:
    unknown = set(methods) - set(METHODS)
    if unknown or set(distributions) - set(DISTRIBUTIONS):
        raise ValueError("Unsupported toxicity evaluation selection")
    selected_distributions = tuple(distributions)
    selected_methods = tuple(method for method in METHODS if method in methods)
    worker_logs = log_root or CACHE / "logs"
    for method in selected_methods:
        if all(
            _generation_path(distribution, method).exists()
            and json.loads(_generation_path(distribution, method).read_text()).get("status")
            == "complete"
            for distribution in selected_distributions
        ):
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage", "generate-final",
            "--model", MODEL_KEY,
            "--method", method,
            "--distributions", ",".join(selected_distributions),
            "--calibration-id", CALIBRATION_ID,
            "--generation-batch-size", str(GENERATION_BATCH_SIZE),
            "--kv-cache", "on" if CURRENT_USE_CACHE else "off",
            "--device", "{device}",
        ]
        run_data_shards(
            f"generate-{method}",
            command,
            devices,
            worker_logs / method,
        )
        for distribution in selected_distributions:
            merge_generation_shards(
                cache_root=CACHE,
                cache_namespace=distribution,
                method=method,
                data_path=DATA_PATH,
                evaluation_key=distribution,
                batch_size=GENERATION_BATCH_SIZE,
                shard_count=len(devices),
            )


def score(
    devices: list[str],
    *,
    methods: list[str] | tuple[str, ...] = METHODS,
    distributions: list[str] | tuple[str, ...] = DISTRIBUTIONS,
    scorers: list[str] | tuple[str, ...] = SCORERS,
    log_root: Path | None = None,
    api_concurrency: int = openai_scoring.DEFAULT_CONCURRENCY,
    api_batch_size: int = openai_scoring.DEFAULT_BATCH_SIZE,
) -> None:
    """Score existing generations without generating model responses."""

    if set(methods) - set(METHODS) or set(distributions) - set(DISTRIBUTIONS):
        raise ValueError("Unsupported toxicity scoring selection")
    if set(scorers) - set(SCORERS):
        raise ValueError("Unsupported toxicity scorer selection")
    selected_methods = tuple(methods)
    selected_scorers = tuple(scorers)
    worker_logs = log_root or CACHE / "logs"
    _launch_workers(
        "score-final", devices, worker_logs,
        candidates=False,
        methods=selected_methods,
        distributions=tuple(distributions),
        scorers=selected_scorers,
        api_concurrency=api_concurrency,
        api_batch_size=api_batch_size,
    )
    for method in selected_methods:
        for distribution in distributions:
            summarize_final(method, distribution, selected_scorers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "calibrate-hinf-base",
            "synthesize-hinf-grid",
            "generate-calibration-worker",
            "score-calibration-grid",
            "select-calibration",
            "calibrate",
            "generate-final",
            "score-final",
            "summarize-final",
            "evaluate",
            "score",
        ),
        required=True,
    )
    parser.add_argument("--model", choices=COMPOSITION.models, required=True)
    parser.add_argument("--calibration-id", default="selected")
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--distribution", choices=DISTRIBUTIONS)
    parser.add_argument("--distributions", default=",".join(DISTRIBUTIONS))
    parser.add_argument("--scorers", default=",".join(SCORERS))
    parser.add_argument(
        "--api-concurrency", type=int, default=openai_scoring.DEFAULT_CONCURRENCY
    )
    parser.add_argument(
        "--api-batch-size", type=int, default=openai_scoring.DEFAULT_BATCH_SIZE
    )
    parser.add_argument("--kv-cache", choices=("off", "on"), default="off")
    arguments = parser.parse_args()
    configure_model(
        arguments.model,
        arguments.calibration_id,
        arguments.generation_batch_size,
        use_cache=arguments.kv_cache == "on",
    )
    selected_distributions = tuple(
        value.strip() for value in arguments.distributions.split(",") if value.strip()
    )
    if not selected_distributions or set(selected_distributions) - set(DISTRIBUTIONS):
        raise ValueError("--distributions contains an unsupported value")
    selected_scorers = tuple(
        value.strip() for value in arguments.scorers.split(",") if value.strip()
    )
    if not selected_scorers or set(selected_scorers) - set(SCORERS):
        raise ValueError("--scorers contains an unsupported value")
    needs_gpu = arguments.stage in {"calibrate", "evaluate"} or (
        arguments.stage == "score"
        and bool({"toxicity_classifier", "perplexity"} & set(selected_scorers))
    )
    devices = resolve_cuda_devices(arguments.devices) if needs_gpu else ["cpu"]
    if arguments.stage == "prepare":
        prepare()
    elif arguments.stage == "calibrate-hinf-base":
        calibrate_hinf_base(arguments.device)
    elif arguments.stage == "synthesize-hinf-grid":
        synthesize_hinf_grid(arguments.device)
    elif arguments.stage == "generate-calibration-worker":
        generate_calibration_worker(
            arguments.device, arguments.worker_index, arguments.worker_count
        )
    elif arguments.stage == "score-calibration-grid":
        score_calibration_grid(
            api_concurrency=arguments.api_concurrency,
            api_batch_size=arguments.api_batch_size,
        )
    elif arguments.stage == "select-calibration":
        select_calibration()
    elif arguments.stage == "calibrate":
        calibrate(
            devices,
            api_concurrency=arguments.api_concurrency,
            api_batch_size=arguments.api_batch_size,
        )
    elif arguments.stage == "generate-final":
        generate_final(
            arguments.method,
            arguments.device,
            selected_distributions,
            arguments.shard_index,
            arguments.shard_count,
        )
    elif arguments.stage == "score-final":
        score_final(
            arguments.method,
            arguments.device,
            selected_distributions,
            selected_scorers,
            api_concurrency=arguments.api_concurrency,
            api_batch_size=arguments.api_batch_size,
        )
    elif arguments.stage == "summarize-final":
        summarize_final(arguments.method, arguments.distribution, selected_scorers)
    elif arguments.stage == "evaluate":
        evaluate(devices)
    elif arguments.stage == "score":
        score(
            devices,
            distributions=selected_distributions,
            scorers=selected_scorers,
            api_concurrency=arguments.api_concurrency,
            api_batch_size=arguments.api_batch_size,
        )
    else:
        raise ValueError(f"Unsupported stage {arguments.stage!r}")


if __name__ == "__main__":
    main()
