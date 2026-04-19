"""Configuration dataclasses for the stage-1 experiment workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem roots for prepared data and experiment outputs."""

    prepared_root: Path = Path("prepared")
    partition_root: Path = Path("datasets")
    centralized_root: Path = Path("centralized")
    federated_root: Path = Path("federated")
    comparison_root: Path = Path("comparisons")
    cache_dir: Path = Path("data/cache/openml")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Global preprocessing settings shared by centralized and federated runs."""

    global_eval_size: float = 0.2
    client_test_size: float = 0.2
    numeric_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "most_frequent"


@dataclass(frozen=True)
class PartitionConfig:
    """Controls client partitioning from the processed global training pool."""

    num_clients: int
    alpha: float
    min_client_samples: int = 10
    max_retries: int = 50


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Shared logistic-regression hyperparameters."""

    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.05
    l2_regularization: float = 0.0


@dataclass(frozen=True)
class DataPreparationConfig:
    """Configuration for building the prepared-data artifacts."""

    dataset_name: str
    seed: int = 42
    paths: ArtifactPaths = field(default_factory=ArtifactPaths)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    partition: PartitionConfig = field(
        default_factory=lambda: PartitionConfig(num_clients=10, alpha=1.0)
    )

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Base configuration shared by centralized and federated experiments."""

    dataset_name: str
    seed: int = 42
    paths: ArtifactPaths = field(default_factory=ArtifactPaths)
    model: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True)
class CentralizedTrainingConfig(ExperimentConfig):
    """Configuration for the centralized baseline."""

    evaluate_on_pooled_client_test: bool = True


@dataclass(frozen=True)
class FederatedTrainingConfig(ExperimentConfig):
    """Configuration for the federated baseline."""

    num_clients: int = 10
    alpha: float = 1.0
    rounds: int = 10
    fit_fraction: float = 1.0
    evaluate_fraction: float = 1.0
    min_available_clients: int = 2
    simulation_backend: str = "auto"
    debug_fallback_on_error: bool = False
    simulation_resources: dict[str, float] = field(
        default_factory=lambda: {"num_cpus": 1.0}
    )


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for centralized-versus-federated comparison reporting."""

    dataset_name: str
    seed: int
    num_clients: int
    alpha: float
    paths: ArtifactPaths = field(default_factory=ArtifactPaths)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


def _serialize_dataclass(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    return _stringify_paths(payload)


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value
