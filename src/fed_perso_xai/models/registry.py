"""Model registry and factory helpers for predictive experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from fed_perso_xai.models.base import TabularClassifier
from fed_perso_xai.models.logistic_regression import LogisticRegressionModel, initialize_parameters
from fed_perso_xai.utils.config import LogisticRegressionConfig


ModelBuilder = Callable[[int, Any], TabularClassifier]
ParameterInitializer = Callable[[int, Any], list[np.ndarray]]


@dataclass(frozen=True)
class ModelSpec:
    """Declarative model factory entry."""

    key: str
    display_name: str
    config_type: type[Any]
    build_model: ModelBuilder
    initialize_parameters: ParameterInitializer


class ModelRegistry:
    """Registry of supported predictive models."""

    def __init__(self, specs: list[ModelSpec] | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ModelSpec) -> None:
        if spec.key in self._specs:
            raise ValueError(f"Model '{spec.key}' is already registered.")
        self._specs[spec.key] = spec

    def get(self, key: str) -> ModelSpec:
        try:
            return self._specs[key]
        except KeyError as exc:
            supported = ", ".join(sorted(self._specs))
            raise ValueError(f"Unsupported model '{key}'. Supported models: {supported}.") from exc

    def list_keys(self) -> list[str]:
        return sorted(self._specs)


def _build_logistic_regression_model(
    n_features: int,
    config: LogisticRegressionConfig,
) -> LogisticRegressionModel:
    return LogisticRegressionModel(
        n_features=n_features,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        local_epochs=config.epochs,
        l2_regularization=config.l2_regularization,
    )


DEFAULT_MODEL_REGISTRY = ModelRegistry(
    specs=[
        ModelSpec(
            key="logistic_regression",
            display_name="Logistic Regression",
            config_type=LogisticRegressionConfig,
            build_model=_build_logistic_regression_model,
            initialize_parameters=lambda n_features, config: initialize_parameters(n_features),
        )
    ]
)


def create_model(
    model_name: str,
    *,
    n_features: int,
    config: Any,
    registry: ModelRegistry | None = None,
) -> TabularClassifier:
    """Build one configured model instance from the registry."""

    spec = (registry or DEFAULT_MODEL_REGISTRY).get(model_name)
    if not isinstance(config, spec.config_type):
        raise TypeError(
            f"Model '{model_name}' expects config type {spec.config_type.__name__}, "
            f"received {type(config).__name__}."
    )
    return spec.build_model(n_features, config)


def build_model_config(
    model_name: str,
    payload: dict[str, Any] | None = None,
    *,
    registry: ModelRegistry | None = None,
) -> Any:
    """Build the typed config object for one model from serialized values."""

    spec = (registry or DEFAULT_MODEL_REGISTRY).get(model_name)
    return spec.config_type(**dict(payload or {}))


def initialize_model_parameters(
    model_name: str,
    *,
    n_features: int,
    config: Any,
    registry: ModelRegistry | None = None,
) -> list[np.ndarray]:
    """Return initial parameters for one configured model."""

    spec = (registry or DEFAULT_MODEL_REGISTRY).get(model_name)
    if not isinstance(config, spec.config_type):
        raise TypeError(
            f"Model '{model_name}' expects config type {spec.config_type.__name__}, "
            f"received {type(config).__name__}."
        )
    return spec.initialize_parameters(n_features, config)
