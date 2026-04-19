from __future__ import annotations

import numpy as np

from fed_perso_xai.fl.strategy import StrategyRegistry, StrategySpec, create_strategy_factory
from fed_perso_xai.utils.config import FederatedTrainingConfig


class DummyStrategyFactory:
    def __init__(self, training_config: FederatedTrainingConfig) -> None:
        self.training_config = training_config

    def create(
        self,
        initial_parameters: list[np.ndarray],
        recorder,
    ) -> dict[str, object]:
        return {
            "rounds": self.training_config.rounds,
            "parameter_count": len(initial_parameters),
            "backend": recorder.backend,
        }


def test_custom_strategy_registry_entry_builds_factory() -> None:
    registry = StrategyRegistry()
    registry.register(
        StrategySpec(
            key="dummy_strategy",
            display_name="Dummy Strategy",
            build_factory=lambda training_config: DummyStrategyFactory(training_config),
        )
    )

    factory = create_strategy_factory(
        "dummy_strategy",
        training_config=FederatedTrainingConfig(dataset_name="adult_income", rounds=3),
        registry=registry,
    )

    assert isinstance(factory, DummyStrategyFactory)
    assert factory.training_config.rounds == 3
    assert registry.list_keys() == ["dummy_strategy"]
