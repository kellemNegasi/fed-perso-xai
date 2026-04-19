from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fed_perso_xai.models.registry import ModelRegistry, ModelSpec, create_model, initialize_model_parameters


@dataclass(frozen=True)
class DummyModelConfig:
    scale: float = 1.0


class DummyModel:
    def __init__(self, n_features: int, config: DummyModelConfig) -> None:
        self.weights = np.full(n_features, config.scale, dtype=np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> float:
        return 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], 0.5, dtype=np.float64)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.ones(X.shape[0], dtype=np.int64)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return 0.0

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights)
        return path

    def get_parameters(self) -> list[np.ndarray]:
        return [self.weights.copy()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.weights = np.asarray(parameters[0], dtype=np.float64)


def test_custom_model_registry_entry_builds_and_initializes() -> None:
    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            key="dummy",
            display_name="Dummy",
            config_type=DummyModelConfig,
            build_model=lambda n_features, config: DummyModel(n_features, config),
            initialize_parameters=lambda n_features, config: [
                np.full(n_features, config.scale, dtype=np.float64)
            ],
        )
    )

    model = create_model(
        "dummy",
        n_features=3,
        config=DummyModelConfig(scale=2.5),
        registry=registry,
    )
    parameters = initialize_model_parameters(
        "dummy",
        n_features=3,
        config=DummyModelConfig(scale=2.5),
        registry=registry,
    )

    assert model.get_parameters()[0].tolist() == [2.5, 2.5, 2.5]
    assert parameters[0].tolist() == [2.5, 2.5, 2.5]
    assert registry.list_keys() == ["dummy"]
