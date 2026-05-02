"""Configuration dataclasses for the baseline experiment workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RECOMMENDER_TYPE = "svm_rank"
_SUPPORTED_RECOMMENDER_TYPES = ("svm_rank", "pairwise_logistic")
_SUPPORTED_RECOMMENDER_CLUSTERING_METHODS = ("secure_kmeans",)


def _normalize_recommender_type(recommender_type: str) -> str:
    normalized = str(recommender_type).strip().lower()
    if normalized not in _SUPPORTED_RECOMMENDER_TYPES:
        supported = ", ".join(_SUPPORTED_RECOMMENDER_TYPES)
        raise ValueError(
            f"Unsupported recommender_type {recommender_type!r}. Supported values: {supported}."
        )
    return normalized


def _normalize_recommender_clustering_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in _SUPPORTED_RECOMMENDER_CLUSTERING_METHODS:
        supported = ", ".join(_SUPPORTED_RECOMMENDER_CLUSTERING_METHODS)
        raise ValueError(
            f"Unsupported clustering.method {method!r}. Supported values: {supported}."
        )
    return normalized


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem roots for prepared data and experiment outputs."""

    prepared_root: Path = Path("prepared")
    partition_root: Path = Path("datasets")
    centralized_root: Path = Path("centralized")
    federated_root: Path = Path("federated")
    comparison_root: Path = Path("comparisons")
    cache_dir: Path = Path("data/cache/openml")

    def __post_init__(self) -> None:
        for field_name in (
            "prepared_root",
            "partition_root",
            "centralized_root",
            "federated_root",
            "comparison_root",
            "cache_dir",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                raise TypeError(f"{field_name} must be a pathlib.Path instance.")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Global preprocessing settings shared by centralized and federated runs."""

    global_eval_size: float = 0.2
    client_test_size: float = 0.2
    fitting_mode: str = "global_shared"
    numeric_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "most_frequent"

    def __post_init__(self) -> None:
        _require_probability("global_eval_size", self.global_eval_size)
        _require_probability("client_test_size", self.client_test_size)
        if self.fitting_mode != "global_shared":
            raise ValueError(
                "fitting_mode must be 'global_shared'. Future preprocessing modes are not implemented yet."
            )


@dataclass(frozen=True)
class PartitionConfig:
    """Controls client partitioning from the processed global training pool."""

    num_clients: int
    alpha: float
    min_client_samples: int = 10
    max_retries: int = 50

    def __post_init__(self) -> None:
        _require_integer_at_least("num_clients", self.num_clients, minimum=2)
        _require_positive("alpha", self.alpha)
        _require_integer_at_least("min_client_samples", self.min_client_samples, minimum=1)
        _require_integer_at_least("max_retries", self.max_retries, minimum=1)


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Shared logistic-regression hyperparameters."""

    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.05
    l2_regularization: float = 0.0

    def __post_init__(self) -> None:
        _require_integer_at_least("epochs", self.epochs, minimum=1)
        _require_integer_at_least("batch_size", self.batch_size, minimum=1)
        _require_positive("learning_rate", self.learning_rate)
        _require_non_negative("l2_regularization", self.l2_regularization)


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

    def __post_init__(self) -> None:
        _require_non_empty_string("dataset_name", self.dataset_name)
        _require_non_negative_integer("seed", self.seed)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Base configuration shared by centralized and federated experiments."""

    dataset_name: str
    seed: int = 42
    model_name: str = "logistic_regression"
    paths: ArtifactPaths = field(default_factory=ArtifactPaths)
    model: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)

    def __post_init__(self) -> None:
        _require_non_empty_string("dataset_name", self.dataset_name)
        _require_non_negative_integer("seed", self.seed)
        _require_non_empty_string("model_name", self.model_name)
        from fed_perso_xai.models.registry import DEFAULT_MODEL_REGISTRY

        DEFAULT_MODEL_REGISTRY.get(self.model_name)

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
    strategy_name: str = "fedavg"
    rounds: int = 10
    fit_fraction: float = 1.0
    evaluate_fraction: float = 1.0
    prediction_threshold: float = 0.5
    min_available_clients: int = 2
    simulation_backend: str = "auto"
    debug_fallback_on_error: bool = False
    ray_num_cpus: int = 4
    secure_aggregation: bool = False
    secure_num_helpers: int = 5
    secure_privacy_threshold: int = 2
    secure_reconstruction_threshold: int | None = None
    secure_field_modulus: int = 2_147_483_647
    secure_quantization_scale: int = 1 << 16
    secure_seed: int = 0
    simulation_resources: dict[str, float] = field(
        default_factory=lambda: {"num_cpus": 1.0, "num_gpus": 0.0}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_integer_at_least("num_clients", self.num_clients, minimum=2)
        _require_positive("alpha", self.alpha)
        _require_non_empty_string("strategy_name", self.strategy_name)
        _require_integer_at_least("rounds", self.rounds, minimum=1)
        _require_fraction_or_one("fit_fraction", self.fit_fraction)
        _require_fraction_or_one("evaluate_fraction", self.evaluate_fraction)
        _require_fraction_or_one("prediction_threshold", self.prediction_threshold)
        _require_integer_at_least("min_available_clients", self.min_available_clients, minimum=1)
        _require_integer_at_least("ray_num_cpus", self.ray_num_cpus, minimum=1)
        _require_integer_at_least("secure_num_helpers", self.secure_num_helpers, minimum=1)
        _require_integer_at_least(
            "secure_privacy_threshold",
            self.secure_privacy_threshold,
            minimum=0,
        )
        if self.secure_reconstruction_threshold is not None:
            _require_integer_at_least(
                "secure_reconstruction_threshold",
                self.secure_reconstruction_threshold,
                minimum=self.secure_privacy_threshold + 1,
            )
            if self.secure_reconstruction_threshold > self.secure_num_helpers:
                raise ValueError(
                    "secure_reconstruction_threshold cannot exceed secure_num_helpers."
                )
        if self.secure_privacy_threshold + 1 > self.secure_num_helpers:
            raise ValueError(
                "secure_num_helpers must be at least secure_privacy_threshold + 1."
            )
        _require_integer_at_least("secure_field_modulus", self.secure_field_modulus, minimum=3)
        _require_integer_at_least(
            "secure_quantization_scale",
            self.secure_quantization_scale,
            minimum=1,
        )
        _require_non_negative_integer("secure_seed", self.secure_seed)
        _validate_simulation_resources(self.simulation_resources)

        from fed_perso_xai.fl.strategy import DEFAULT_STRATEGY_REGISTRY

        DEFAULT_STRATEGY_REGISTRY.get(self.strategy_name)


@dataclass(frozen=True)
class RecommenderFederatedTrainingConfig:
    """Configuration for federated pairwise explanation-recommender training."""

    run_id: str
    selection_id: str
    persona: str
    recommender_type: str = DEFAULT_RECOMMENDER_TYPE
    paths: ArtifactPaths = field(default_factory=ArtifactPaths)
    rounds: int = 10
    strategy_name: str = "fedavg"
    fit_fraction: float = 1.0
    evaluate_fraction: float = 1.0
    min_available_clients: int = 2
    simulation_backend: str = "auto"
    debug_fallback_on_error: bool = False
    ray_num_cpus: int = 4
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.05
    l2_regularization: float = 0.0
    seed: int = 42
    top_k: tuple[int, ...] = (1, 3, 5, 8)
    context_filename: str = "candidate_context.parquet"
    label_filename: str = "pairwise_labels.parquet"
    clients: str = "all"
    runtime_num_clients: int = 0
    secure_aggregation: bool = False
    secure_num_helpers: int = 5
    secure_privacy_threshold: int = 2
    secure_reconstruction_threshold: int | None = None
    secure_field_modulus: int = 2_147_483_647
    secure_quantization_scale: int = 1 << 16
    secure_seed: int = 0
    clustering: "RecommenderClusteringConfig" = field(
        default_factory=lambda: RecommenderClusteringConfig()
    )
    simulation_resources: dict[str, float] = field(
        default_factory=lambda: {"num_cpus": 1.0, "num_gpus": 0.0}
    )

    def __post_init__(self) -> None:
        _require_non_empty_string("run_id", self.run_id)
        _require_non_empty_string("selection_id", self.selection_id)
        _require_non_empty_string("persona", self.persona)
        _normalize_recommender_type(self.recommender_type)
        _require_integer_at_least("rounds", self.rounds, minimum=1)
        _require_non_empty_string("strategy_name", self.strategy_name)
        _require_fraction_or_one("fit_fraction", self.fit_fraction)
        _require_fraction_or_one("evaluate_fraction", self.evaluate_fraction)
        _require_integer_at_least("min_available_clients", self.min_available_clients, minimum=1)
        _require_integer_at_least("ray_num_cpus", self.ray_num_cpus, minimum=1)
        _require_integer_at_least("epochs", self.epochs, minimum=1)
        _require_integer_at_least("batch_size", self.batch_size, minimum=1)
        _require_positive("learning_rate", self.learning_rate)
        _require_non_negative("l2_regularization", self.l2_regularization)
        _require_non_negative_integer("seed", self.seed)
        _require_non_empty_string("context_filename", self.context_filename)
        _require_non_empty_string("label_filename", self.label_filename)
        _require_non_empty_string("clients", self.clients)
        _require_non_negative_integer("runtime_num_clients", self.runtime_num_clients)
        if not self.top_k:
            raise ValueError("top_k must contain at least one value.")
        for value in self.top_k:
            _require_integer_at_least("top_k item", int(value), minimum=1)
        _require_integer_at_least("secure_num_helpers", self.secure_num_helpers, minimum=1)
        _require_integer_at_least(
            "secure_privacy_threshold",
            self.secure_privacy_threshold,
            minimum=0,
        )
        if self.secure_reconstruction_threshold is not None:
            _require_integer_at_least(
                "secure_reconstruction_threshold",
                self.secure_reconstruction_threshold,
                minimum=self.secure_privacy_threshold + 1,
            )
            if self.secure_reconstruction_threshold > self.secure_num_helpers:
                raise ValueError(
                    "secure_reconstruction_threshold cannot exceed secure_num_helpers."
                )
        if self.secure_privacy_threshold + 1 > self.secure_num_helpers:
            raise ValueError(
                "secure_num_helpers must be at least secure_privacy_threshold + 1."
            )
        _require_integer_at_least("secure_field_modulus", self.secure_field_modulus, minimum=3)
        _require_integer_at_least(
            "secure_quantization_scale",
            self.secure_quantization_scale,
            minimum=1,
        )
        _require_non_negative_integer("secure_seed", self.secure_seed)
        if not isinstance(self.clustering, RecommenderClusteringConfig):
            raise TypeError("clustering must be a RecommenderClusteringConfig instance.")
        if self.runtime_num_clients > 0 and self.clustering.enabled:
            if self.clustering.k > self.runtime_num_clients:
                raise ValueError(
                    "clustering.k cannot exceed the loaded recommender client count."
                )
        if self.clustering.enabled and self.clustering.warmup_rounds >= self.rounds:
            raise ValueError(
                "clustering.warmup_rounds must be smaller than rounds when clustered training is enabled."
            )
        _validate_simulation_resources(self.simulation_resources)

        from fed_perso_xai.fl.strategy import DEFAULT_STRATEGY_REGISTRY

        DEFAULT_STRATEGY_REGISTRY.get(self.strategy_name)

    @property
    def num_clients(self) -> int:
        """Client count used by the shared FedAvg strategy helpers."""

        return int(self.runtime_num_clients or self.min_available_clients)

    def with_num_clients(self, num_clients: int) -> "RecommenderFederatedTrainingConfig":
        """Return a copy with the loaded client count attached."""

        _require_integer_at_least("num_clients", num_clients, minimum=1)
        return RecommenderFederatedTrainingConfig(
            run_id=self.run_id,
            selection_id=self.selection_id,
            persona=self.persona,
            recommender_type=self.recommender_type,
            paths=self.paths,
            rounds=self.rounds,
            strategy_name=self.strategy_name,
            fit_fraction=self.fit_fraction,
            evaluate_fraction=self.evaluate_fraction,
            min_available_clients=self.min_available_clients,
            simulation_backend=self.simulation_backend,
            debug_fallback_on_error=self.debug_fallback_on_error,
            ray_num_cpus=self.ray_num_cpus,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            seed=self.seed,
            top_k=tuple(self.top_k),
            context_filename=self.context_filename,
            label_filename=self.label_filename,
            clients=self.clients,
            runtime_num_clients=int(num_clients),
            secure_aggregation=self.secure_aggregation,
            secure_num_helpers=self.secure_num_helpers,
            secure_privacy_threshold=self.secure_privacy_threshold,
            secure_reconstruction_threshold=self.secure_reconstruction_threshold,
            secure_field_modulus=self.secure_field_modulus,
            secure_quantization_scale=self.secure_quantization_scale,
            secure_seed=self.secure_seed,
            clustering=self.clustering,
            simulation_resources=dict(self.simulation_resources),
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True)
class RecommenderClusteringConfig:
    """Optional clustered aggregation for recommender federated training."""

    enabled: bool = False
    method: str = "secure_kmeans"
    k: int = 3
    enable_pca: bool = True
    pca_components: int = 8
    warmup_rounds: int = 0
    freeze_pca_after_warmup: bool = False
    max_iterations: int = 20
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean.")
        _normalize_recommender_clustering_method(self.method)
        _require_integer_at_least("k", self.k, minimum=1)
        if not isinstance(self.enable_pca, bool):
            raise TypeError("enable_pca must be a boolean.")
        _require_integer_at_least("pca_components", self.pca_components, minimum=1)
        _require_non_negative_integer("warmup_rounds", self.warmup_rounds)
        if not isinstance(self.freeze_pca_after_warmup, bool):
            raise TypeError("freeze_pca_after_warmup must be a boolean.")
        _require_integer_at_least("max_iterations", self.max_iterations, minimum=1)
        _require_positive("tolerance", self.tolerance)


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for centralized-versus-federated comparison reporting."""

    dataset_name: str
    seed: int
    num_clients: int
    alpha: float
    paths: ArtifactPaths = field(default_factory=ArtifactPaths)

    def __post_init__(self) -> None:
        _require_non_empty_string("dataset_name", self.dataset_name)
        _require_non_negative_integer("seed", self.seed)
        _require_integer_at_least("num_clients", self.num_clients, minimum=2)
        _require_positive("alpha", self.alpha)

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


def _require_non_empty_string(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_integer_at_least(field_name: str, value: int, *, minimum: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")


def _require_non_negative_integer(field_name: str, value: int) -> None:
    _require_integer_at_least(field_name, value, minimum=0)


def _require_positive(field_name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    if float(value) <= 0.0:
        raise ValueError(f"{field_name} must be strictly positive.")


def _require_non_negative(field_name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    if float(value) < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")


def _validate_simulation_resources(simulation_resources: dict[str, float]) -> None:
    if not isinstance(simulation_resources, dict):
        raise TypeError("simulation_resources must be a dictionary.")
    for resource_name, value in simulation_resources.items():
        field_name = f"simulation_resources[{resource_name!r}]"
        if resource_name == "num_gpus":
            _require_non_negative(field_name, value)
            continue
        _require_positive(field_name, value)


def _require_probability(field_name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{field_name} must be strictly between 0 and 1.")


def _require_fraction_or_one(field_name: str, value: float) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be in the interval [0, 1].")
