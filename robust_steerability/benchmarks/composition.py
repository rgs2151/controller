"""Declarative composition of one fitted behavior with evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass
import tomllib

from robust_steerability.benchmarks.layout import benchmark_root
from robust_steerability.benchmarks.methods import validate_methods
from robust_steerability.judges.specs import scorer_spec
from robust_steerability.source_methods.protocol import GENERATION


ROLES = {"base", "transfer", "capability"}


@dataclass(frozen=True)
class EvaluationDataset:
    key: str
    dataset: str
    role: str
    cache_namespace: str
    generation_profile: str
    samples: int
    repetitions: int
    scorers: tuple[str, ...]
    runtime: str


@dataclass(frozen=True)
class HInfinityLambdaSweep:
    """Optional first phase of H-infinity hyperparameter calibration."""

    enabled: bool
    values: tuple[float, ...]
    fixed_q_over_r: float
    fixed_q_final_over_r: float
    fixed_r: float
    selection_metric: str


@dataclass(frozen=True)
class CalibrationComposition:
    selection_metric: str
    available_selection_metrics: tuple[str, ...]
    h_infinity_lambda_sweep: HInfinityLambdaSweep


@dataclass(frozen=True)
class BenchmarkRunProfile:
    """A named, explicitly scoped benchmark invocation."""

    name: str
    models: tuple[str, ...]
    methods: tuple[str, ...]
    datasets: tuple[str, ...]
    evaluation_repetitions: int | None = None


@dataclass(frozen=True)
class BenchmarkComposition:
    benchmark: str
    base_dataset: str
    models: tuple[str, ...]
    available_methods: tuple[str, ...]
    default_methods: tuple[str, ...]
    default_datasets: tuple[str, ...]
    datasets: tuple[EvaluationDataset, ...]
    calibration: CalibrationComposition
    run_profiles: tuple[BenchmarkRunProfile, ...]

    def dataset(self, key: str) -> EvaluationDataset:
        matches = [dataset for dataset in self.datasets if dataset.key == key]
        if len(matches) != 1:
            raise ValueError(f"Unknown {self.benchmark} evaluation dataset {key!r}")
        return matches[0]

    def run_profile(self, name: str) -> BenchmarkRunProfile:
        matches = [profile for profile in self.run_profiles if profile.name == name]
        if len(matches) != 1:
            raise ValueError(f"Unknown {self.benchmark} run profile {name!r}")
        return matches[0]

    @property
    def dataset_keys(self) -> tuple[str, ...]:
        return tuple(dataset.key for dataset in self.datasets)


def load_composition(benchmark: str) -> BenchmarkComposition:
    path = benchmark_root(benchmark) / "benchmark.toml"
    payload = tomllib.loads(path.read_text())
    models = tuple(str(value) for value in payload["models"])
    available_methods = tuple(str(value) for value in payload["available_methods"])
    default_methods = tuple(str(value) for value in payload["default_methods"])
    validate_methods(available_methods)
    if not default_methods or set(default_methods) - set(available_methods):
        raise ValueError(
            f"{path} default_methods must be a non-empty subset of available_methods"
        )
    entries = payload.get("evaluation_datasets", [])
    if not entries:
        raise ValueError(f"{path} does not define evaluation_datasets")
    datasets = tuple(
        EvaluationDataset(
            key=str(entry["key"]),
            dataset=str(entry["dataset"]),
            role=str(entry["role"]),
            cache_namespace=str(entry["cache_namespace"]),
            generation_profile=str(entry["generation_profile"]),
            samples=int(entry["samples"]),
            repetitions=int(entry["repetitions"]),
            scorers=tuple(str(value) for value in entry["scorers"]),
            runtime=str(entry["runtime"]),
        )
        for entry in entries
    )
    keys = [dataset.key for dataset in datasets]
    namespaces = [dataset.cache_namespace for dataset in datasets]
    if len(keys) != len(set(keys)) or len(namespaces) != len(set(namespaces)):
        raise ValueError(f"{path} contains duplicate dataset keys or cache namespaces")
    default_datasets = tuple(str(value) for value in payload["default_datasets"])
    if not default_datasets or set(default_datasets) - set(keys):
        raise ValueError(
            f"{path} default_datasets must be a non-empty subset of evaluation datasets"
        )
    if any(dataset.role not in ROLES for dataset in datasets):
        raise ValueError(f"{path} contains an unsupported evaluation role")
    if any(dataset.generation_profile not in GENERATION for dataset in datasets):
        raise ValueError(f"{path} contains an unsupported generation profile")
    if any(dataset.samples < 1 or dataset.repetitions < 1 for dataset in datasets):
        raise ValueError(f"{path} requires positive sample and repetition counts")
    for dataset in datasets:
        for scorer in dataset.scorers:
            scorer_spec(scorer)
    base = [dataset for dataset in datasets if dataset.role == "base"]
    if len(base) != 1 or base[0].dataset != str(payload["base_dataset"]):
        raise ValueError(f"{path} must define exactly one matching base dataset")
    calibration_payload = payload.get("calibration", {})
    selection_metric = str(
        calibration_payload.get("selection_metric", "mean_axbench_overall")
    )
    available_selection_metrics = tuple(
        str(value)
        for value in calibration_payload.get(
            "available_selection_metrics", (selection_metric,)
        )
    )
    if (
        not available_selection_metrics
        or len(available_selection_metrics) != len(set(available_selection_metrics))
        or selection_metric not in available_selection_metrics
    ):
        raise ValueError(
            f"{path} calibration selection metric must appear exactly once in "
            "available_selection_metrics"
        )
    lambda_payload = calibration_payload.get("h_infinity_lambda_sweep", {})
    lambda_enabled = bool(lambda_payload.get("enabled", False))
    lambda_values = tuple(float(value) for value in lambda_payload.get("values", ()))
    lambda_q_over_r = float(lambda_payload.get("fixed_q_over_r", 0.1))
    lambda_q_final_over_r = float(
        lambda_payload.get("fixed_q_final_over_r", 0.1)
    )
    lambda_r = float(lambda_payload.get("fixed_r", 1.0))
    lambda_metric = str(
        lambda_payload.get(
            "selection_metric",
            calibration_payload.get("selection_metric", "mean_axbench_overall"),
        )
    )
    if lambda_enabled and not lambda_values:
        raise ValueError(f"{path} enables the H-infinity lambda sweep without values")
    if any(value <= 0 for value in lambda_values):
        raise ValueError(f"{path} requires positive H-infinity lambda values")
    if len(lambda_values) != len(set(lambda_values)):
        raise ValueError(f"{path} contains duplicate H-infinity lambda values")
    if min(lambda_q_over_r, lambda_q_final_over_r, lambda_r) <= 0:
        raise ValueError(f"{path} requires positive fixed lambda-sweep Q, Qf, and R")
    run_profiles = tuple(
        BenchmarkRunProfile(
            name=str(name),
            models=tuple(str(value) for value in profile["models"]),
            methods=tuple(str(value) for value in profile["methods"]),
            datasets=tuple(str(value) for value in profile["datasets"]),
            evaluation_repetitions=(
                int(profile["evaluation_repetitions"])
                if "evaluation_repetitions" in profile
                else None
            ),
        )
        for name, profile in payload.get("run_profiles", {}).items()
    )
    for profile in run_profiles:
        if not profile.models or set(profile.models) - set(models):
            raise ValueError(f"{path} run profile {profile.name!r} has invalid models")
        if not profile.methods or set(profile.methods) - set(available_methods):
            raise ValueError(f"{path} run profile {profile.name!r} has invalid methods")
        if not profile.datasets or set(profile.datasets) - set(keys):
            raise ValueError(f"{path} run profile {profile.name!r} has invalid datasets")
        if (
            profile.evaluation_repetitions is not None
            and profile.evaluation_repetitions < 1
        ):
            raise ValueError(
                f"{path} run profile {profile.name!r} requires positive repetitions"
            )
    return BenchmarkComposition(
        benchmark=benchmark,
        base_dataset=str(payload["base_dataset"]),
        models=models,
        available_methods=available_methods,
        default_methods=default_methods,
        default_datasets=default_datasets,
        datasets=datasets,
        calibration=CalibrationComposition(
            selection_metric=selection_metric,
            available_selection_metrics=available_selection_metrics,
            h_infinity_lambda_sweep=HInfinityLambdaSweep(
                enabled=lambda_enabled,
                values=lambda_values,
                fixed_q_over_r=lambda_q_over_r,
                fixed_q_final_over_r=lambda_q_final_over_r,
                fixed_r=lambda_r,
                selection_metric=lambda_metric,
            ),
        ),
        run_profiles=run_profiles,
    )


def requested_scorers(
    dataset: EvaluationDataset,
    requested: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if requested is None:
        return dataset.scorers
    return tuple(scorer for scorer in requested if scorer in dataset.scorers)


def validate_requested_scorers(
    composition: BenchmarkComposition,
    datasets: list[str],
    requested: tuple[str, ...] | None,
) -> None:
    if requested is None:
        return
    available = {
        scorer
        for dataset_key in datasets
        for scorer in composition.dataset(dataset_key).scorers
    }
    invalid = set(requested) - available
    if invalid:
        raise ValueError(
            f"Scorers {sorted(invalid)} do not apply to the selected datasets; "
            f"available scorers are {sorted(available)}"
        )
