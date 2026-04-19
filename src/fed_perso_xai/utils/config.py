"""Configuration dataclasses for stage-1 pipeline components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreprocessingConfig:
    """Global preprocessing settings shared by all clients."""

    global_eval_size: float = 0.2
    client_test_size: float = 0.2
    numeric_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "most_frequent"


@dataclass(frozen=True)
class PartitionConfig:
    """Controls client creation from the processed global training pool."""

    num_clients: int
    alpha: float
    min_client_samples: int = 10
    max_retries: int = 50


@dataclass(frozen=True)
class DataPreparationConfig:
    """Top-level dataset preparation configuration."""

    dataset_name: str
    seed: int = 42
    output_root: Path = Path("datasets")
    cache_dir: Path = Path("data/cache/openml")
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    partition: PartitionConfig = field(
        default_factory=lambda: PartitionConfig(num_clients=10, alpha=1.0)
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["cache_dir"] = str(self.cache_dir)
        return payload


@dataclass(frozen=True)
class FederatedTrainingConfig:
    """Training and evaluation settings for the federated baseline."""

    dataset_name: str
    seed: int = 42
    data_root: Path = Path("datasets")
    results_root: Path = Path("results")
    num_clients: int = 10
    alpha: float = 1.0
    rounds: int = 10
    local_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.05
    l2_regularization: float = 0.0
    fit_fraction: float = 1.0
    evaluate_fraction: float = 1.0
    min_available_clients: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["results_root"] = str(self.results_root)
        return payload
