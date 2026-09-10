"""Frozen protocols for the non-H-infinity methods in the A-LQR paper.

These values are transcribed from the preserved A-LQR implementation and the
baseline adapters that were present immediately before commit 626f757 removed
them from the public tree.  Evaluation sample count is deliberately absent
from every calibration and tuning object: callers choose 50 for the pilot and
the paper-sized count later without changing the methods.
"""

from __future__ import annotations

from dataclasses import dataclass


ALQR_SOURCE_REVISION = "c2e0c8450797e0dc234b2c53475267a2cfb2457a"
ALQR_BASELINE_ADAPTER_REVISION = "11b49bad15c02f0a23d9481980d4b88f2d6313a5"
UPSTREAM_REVISIONS = {
    "iti": "2c6b2179be7b5aa8f0a171688cf9e01b812ca327",
    "mean_act_linear_act": "d2c3560b7022da795d58f892c398ab77cff13590",
    "pid_act": "d705c44a2f9c67e54e4d551824f46cba3c3b187e",
    "odesteer": "8a3c481d6493ecb3325eea5ef9c448cccfced7eb",
}


@dataclass(frozen=True)
class CalibrationCounts:
    negative: int
    positive: int
    jacobian: int
    jacobian_class: str
    jacobian_max_length: int = 24


@dataclass(frozen=True)
class LQRSweep:
    lambdas: tuple[float, ...]
    q: float
    r: float
    q_final: float


@dataclass(frozen=True)
class PIDSweep:
    lambdas: tuple[float, ...]
    kp: float
    ki: float
    kd: float


MODEL_IDS = {
    "llama1b": "meta-llama/Llama-3.2-1B",
    "gemma2b": "google/gemma-2-2b",
    "qwen3b": "Qwen/Qwen2.5-3B",
    "llama8b": "meta-llama/Meta-Llama-3-8B",
    "gemma9b": "google/gemma-2-9b",
    "qwen14b": "Qwen/Qwen2.5-14B",
    "qwen32b": "Qwen/Qwen2.5-32B",
}
MODEL_KEYS = {model_id: key for key, model_id in MODEL_IDS.items()}


CALIBRATION_COUNTS = {
    "toxicity": CalibrationCounts(negative=200, positive=200, jacobian=50, jacobian_class="positive"),
    "truthfulness": CalibrationCounts(negative=12, positive=12, jacobian=1, jacobian_class="positive"),
}


ALQR_SWEEPS = {
    "toxicity": {
        "gemma2b": LQRSweep((1.0, 1.25, 1.375, 1.5), 0.1, 3.0, 0.1),
        "llama8b": LQRSweep((2.0, 2.5), 0.1, 10.0, 10.0),
        "gemma9b": LQRSweep((2.0,), 0.1, 1.0, 0.1),
        "qwen14b": LQRSweep((2.0, 2.5), 1.0, 1.0, 1.0),
        "qwen3b": LQRSweep((3.0,), 0.1, 10.0, 1.0),
        "llama1b": LQRSweep((2.0, 2.5, 3.5), 0.1, 1.0, 1.0),
        "qwen32b": LQRSweep((2.0, 2.5, 3.5), 1.0, 5.0, 0.1),
    },
    "truthfulness": {
        "gemma2b": LQRSweep((3.0,), 0.1, 1.0, 0.3),
        "llama8b": LQRSweep((2.0, 2.5, 3.5), 0.1, 10.0, 10.0),
        "gemma9b": LQRSweep((2.0, 2.5, 3.5), 0.1, 1.0, 0.1),
        "qwen14b": LQRSweep((3.0, 3.5), 0.1, 1.0, 0.3),
        "qwen3b": LQRSweep((3.0, 3.5), 0.1, 1.0, 0.3),
        "llama1b": LQRSweep((2.0, 2.5, 3.5), 0.1, 1.0, 1.0),
        "qwen32b": LQRSweep((2.0, 2.5, 3.5), 1.0, 5.0, 0.1),
    },
}


SPID_SWEEPS = {
    "toxicity": {
        "gemma2b": PIDSweep((0.5, 1.0), 0.7, 0.01, 0.1),
        "llama8b": PIDSweep((0.5, 1.0, 1.5), 0.1, 0.1, 0.0),
        "gemma9b": PIDSweep((0.5, 1.0), 0.7, 0.05, 0.0),
        "qwen14b": PIDSweep((0.5, 1.0), 0.5, 0.01, 0.01),
        "qwen3b": PIDSweep((0.5, 1.0), 0.3, 0.1, 0.0),
        "llama1b": PIDSweep((1.0,), 0.5, 0.5, 0.1),
        "qwen32b": PIDSweep((1.0, 1.5), 0.7, 0.1, 0.0),
    },
    "truthfulness": {
        "gemma2b": PIDSweep((0.5, 1.0, 1.5), 0.7, 0.01, 0.1),
        "llama8b": PIDSweep((0.5, 1.0, 1.5), 0.1, 0.1, 0.0),
        "gemma9b": PIDSweep((0.5, 1.0, 1.5), 0.7, 0.05, 0.0),
        "qwen14b": PIDSweep((2.0,), 0.5, 0.01, 0.01),
        "qwen3b": PIDSweep((0.5, 1.0, 1.5), 0.3, 0.1, 0.0),
        "llama1b": PIDSweep((0.5, 1.0, 1.5), 0.5, 0.5, 0.1),
        "qwen32b": PIDSweep((1.5,), 0.7, 0.1, 0.0),
    },
}


ACTADD_SWEEP = {
    "layers": "all",
    "strengths": (0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    "fit_samples_per_class": 100,
}
ACTADD_PAPER_SELECTIONS = {
    "llama1b": (8, 4.0),
    "gemma2b": (12, 4.0),
    "qwen3b": (13, 2.0),
    "llama8b": (13, 4.0),
    "gemma9b": (15, 4.0),
    "qwen14b": (21, 4.0),
}


ITI_SWEEPS = {
    "toxicity": {"top_heads": (16, 32, 64), "alphas": (5.0, 10.0, 20.0)},
    "truthfulness": {"top_heads": (16, 32, 64), "alphas": (5.0, 10.0, 15.0)},
}
ITI_FIT_SAMPLES_PER_CLASS = 80


ACT_SWEEPS = {
    "toxicity": (0.5, 1.0, 2.0),
    "truthfulness": (0.5, 1.0, 1.5),
}
ACT_FIT_SAMPLES_PER_CLASS = 200
ACT_ADAPTER_MODULE_LIMIT = 4
ACT_MODULE_PATTERNS = {
    "llama1b": (
        r"model.layers.*.mlp.up_proj",
        r"model.layers.*.mlp.down_proj",
        r"model.layers.*.mlp.gate_proj",
    ),
    "gemma2b": (r".*post_attention_layernorm", r".*post_feedforward_layernorm"),
    "qwen3b": (
        r"model.layers.*.mlp.up_proj",
        r"model.layers.*.mlp.down_proj",
        r"model.layers.*.mlp.gate_proj",
    ),
    "llama8b": (
        r"model.layers.*.mlp.up_proj",
        r"model.layers.*.mlp.down_proj",
        r"model.layers.*.mlp.gate_proj",
    ),
    "gemma9b": (r".*post_attention_layernorm", r".*post_feedforward_layernorm"),
    "qwen14b": (
        r"model.layers.*.mlp.up_proj",
        r"model.layers.*.mlp.down_proj",
        r"model.layers.*.mlp.gate_proj",
    ),
}


ODESTEER_T_VALUES = (1.0, 5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 65.0, 80.0, 100.0, 120.0, 150.0)
ODESTEER_FIT_SAMPLES = {"toxicity": 5000, "truthfulness": 1000}
ODESTEER_PARAMETERS = {
    "solver": "euler",
    "steps": 10,
    "n_components": 8000,
    "degree": 2,
    "gamma": 0.1,
    "coef0": 1.0,
    "linear_classifier": "logistic_regression",
}


GENERATION = {
    "toxicity": {
        "max_new_tokens": 100, "temperature": 1.0, "top_p": 0.3,
        "repetition_penalty": 1.2, "do_sample": True,
    },
    "truthfulness": {
        "max_new_tokens": 50, "temperature": 1.0, "top_p": 0.3,
        "repetition_penalty": 1.2, "do_sample": True,
    },
}
EVALUATION_REPETITIONS = 5
SOURCE_RANDOM_SEED = 42
GENERATION_CACHE = {"original": False, "steered": True}


METHOD_MODEL_LOADING = {
    "original": {"quantization": "nf4", "double_quantization": True, "model_dtype": "float32"},
    "alqr": {"quantization": "nf4", "double_quantization": True, "model_dtype": "float32"},
    "spid": {"quantization": "nf4", "double_quantization": True, "model_dtype": "float32"},
    "actadd": {
        "quantization": "nf4", "double_quantization": True,
        "model_dtype": "float32", "compute_dtype": "float16",
    },
    "iti": {"quantization": None, "model_dtype": "float16"},
    "mean_act": {"quantization": None, "model_dtype": "bfloat16"},
    "linear_act": {"quantization": None, "model_dtype": "bfloat16"},
    "pid_act": {"quantization": None, "model_dtype": "bfloat16"},
    "odesteer": {
        "quantization": "nf4", "double_quantization": True,
        "model_dtype": "float32", "compute_dtype": "float16",
    },
}


def model_key(model_id: str) -> str:
    """Resolve only checkpoints explicitly supported by the preserved source."""

    if model_id not in MODEL_KEYS:
        raise ValueError(f"No source-defined protocol for model {model_id!r}")
    return MODEL_KEYS[model_id]


def control_sweeps(behavior: str, model_id: str) -> tuple[LQRSweep, PIDSweep]:
    """Return the exact A-LQR and S-PID grids for one source checkpoint."""

    key = model_key(model_id)
    if behavior not in ALQR_SWEEPS:
        raise ValueError(f"Unsupported behavior {behavior!r}")
    return ALQR_SWEEPS[behavior][key], SPID_SWEEPS[behavior][key]


def act_module_patterns(model_id: str) -> tuple[str, ...]:
    """Return only module patterns explicitly present in the comparison adapter."""

    key = model_key(model_id)
    if key not in ACT_MODULE_PATTERNS:
        raise ValueError(f"No source-defined AcT module protocol for model {model_id!r}")
    return ACT_MODULE_PATTERNS[key]


def odesteer_layers(layer_count: int) -> tuple[int, ...]:
    """Return the inclusive middle-half layer sweep used by the adapter."""

    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    return tuple(range(layer_count // 4, (3 * layer_count) // 4 + 1))


def protocol_manifest(behavior: str, model_id: str, checkpoint_revision: str, evaluation_samples: int) -> dict:
    """Create an auditable run manifest without altering calibration settings."""

    if not checkpoint_revision:
        raise ValueError("A concrete Hugging Face checkpoint revision is required")
    if evaluation_samples < 1:
        raise ValueError("evaluation_samples must be positive")
    key = model_key(model_id)
    alqr, spid = control_sweeps(behavior, model_id)
    patterns = act_module_patterns(model_id)
    if key not in ACTADD_PAPER_SELECTIONS:
        raise ValueError(f"No source-defined ActAdd selection for model {model_id!r}")
    return {
        "model_id": model_id,
        "checkpoint_revision": checkpoint_revision,
        "behavior": behavior,
        "evaluation_samples": evaluation_samples,
        "evaluation_repetitions": EVALUATION_REPETITIONS,
        "random_seed": SOURCE_RANDOM_SEED,
        "calibration": CALIBRATION_COUNTS[behavior].__dict__,
        "alqr_sweep": alqr.__dict__,
        "spid_sweep": spid.__dict__,
        "actadd_sweep": ACTADD_SWEEP,
        "actadd_selected": {
            "layer": ACTADD_PAPER_SELECTIONS[key][0],
            "strength": ACTADD_PAPER_SELECTIONS[key][1],
        },
        "iti_sweep": ITI_SWEEPS[behavior],
        "iti_fit": {
            "samples_per_class": ITI_FIT_SAMPLES_PER_CLASS,
            "max_length": 50,
            "validation_fraction": 0.2,
            "stratified": True,
            "seed": 42,
        },
        "act_sweep": ACT_SWEEPS[behavior],
        "act_fit_samples_per_class": ACT_FIT_SAMPLES_PER_CLASS,
        "act_adapter_module_limit": ACT_ADAPTER_MODULE_LIMIT,
        "act_module_patterns": patterns,
        "odesteer": {**ODESTEER_PARAMETERS, "fit_samples": ODESTEER_FIT_SAMPLES[behavior]},
        "generation": GENERATION[behavior],
        "generation_cache": GENERATION_CACHE,
        "method_model_loading": {
            name: {
                **settings,
                **(
                    {"compute_dtype": "float32"}
                    if name in {"original", "alqr", "spid"} and behavior == "truthfulness"
                    else {"compute_dtype": "float16"}
                    if name in {"original", "alqr", "spid"}
                    else {}
                ),
            }
            for name, settings in METHOD_MODEL_LOADING.items()
        },
        "source_revisions": {
            "alqr": ALQR_SOURCE_REVISION,
            "alqr_baseline_adapters": ALQR_BASELINE_ADAPTER_REVISION,
            **UPSTREAM_REVISIONS,
        },
        "sweep_rule": "run and retain every source-defined candidate before selecting a reported row",
    }
