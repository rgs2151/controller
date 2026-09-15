"""Declarative composition of one fitted behavior with evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass
import tomllib

from robust_steerability.benchmarks.layout import benchmark_root
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
class BenchmarkComposition:
    benchmark: str
    base_dataset: str
    datasets: tuple[EvaluationDataset, ...]

    def dataset(self, key: str) -> EvaluationDataset:
        matches = [dataset for dataset in self.datasets if dataset.key == key]
        if len(matches) != 1:
            raise ValueError(f"Unknown {self.benchmark} evaluation dataset {key!r}")
        return matches[0]

    @property
    def dataset_keys(self) -> tuple[str, ...]:
        return tuple(dataset.key for dataset in self.datasets)


def load_composition(benchmark: str) -> BenchmarkComposition:
    path = benchmark_root(benchmark) / "benchmark.toml"
    payload = tomllib.loads(path.read_text())
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
    return BenchmarkComposition(
        benchmark=benchmark,
        base_dataset=str(payload["base_dataset"]),
        datasets=datasets,
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
