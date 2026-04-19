"""Flower strategy helpers with tracking and future extension seams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

FLOWER_IMPORT_ERROR_MESSAGE = (
    "Flower support is not installed. Install the optional federated extras with "
    "`pip install -e .[fl]` for debug runtime support or `pip install -e .[ray]` "
    "for Ray-backed simulation."
)

try:
    import flwr as fl
except ImportError:  # pragma: no cover - exercised via optional dependency paths
    fl = None  # type: ignore[assignment]

from fed_perso_xai.evaluation.metrics import aggregate_weighted_metrics
from fed_perso_xai.utils.config import FederatedTrainingConfig


@dataclass
class FederatedRunRecorder:
    """Mutable recorder populated by the Flower strategy during training."""

    backend: str
    round_history: list[dict[str, object]] = field(default_factory=list)
    final_parameters: list[np.ndarray] | None = None


class StrategyFactory(Protocol):
    """Protocol for pluggable aggregation strategies in later stages."""

    def create(
        self,
        initial_parameters: list[np.ndarray],
        recorder: FederatedRunRecorder,
        ) -> Any:
        """Build a Flower strategy."""


StrategyFactoryBuilder = Callable[[FederatedTrainingConfig], StrategyFactory]


@dataclass(frozen=True)
class StrategySpec:
    """Declarative strategy-factory entry."""

    key: str
    display_name: str
    build_factory: StrategyFactoryBuilder


class StrategyRegistry:
    """Registry of supported federated aggregation strategies."""

    def __init__(self, specs: list[StrategySpec] | None = None) -> None:
        self._specs: dict[str, StrategySpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: StrategySpec) -> None:
        if spec.key in self._specs:
            raise ValueError(f"Strategy '{spec.key}' is already registered.")
        self._specs[spec.key] = spec

    def get(self, key: str) -> StrategySpec:
        try:
            return self._specs[key]
        except KeyError as exc:
            supported = ", ".join(sorted(self._specs))
            raise ValueError(
                f"Unsupported strategy '{key}'. Supported strategies: {supported}."
            ) from exc

    def list_keys(self) -> list[str]:
        return sorted(self._specs)


if fl is not None:

    class TrackingFedAvg(fl.server.strategy.FedAvg):
        """FedAvg strategy that records round history and final parameters."""

        def __init__(self, *args, recorder: FederatedRunRecorder, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.recorder = recorder

        def aggregate_fit(self, server_round, results, failures):  # type: ignore[override]
            parameters, metrics = super().aggregate_fit(server_round, results, failures)
            while len(self.recorder.round_history) < server_round:
                self.recorder.round_history.append({"round": len(self.recorder.round_history) + 1})
            self.recorder.round_history[server_round - 1]["fit_metrics"] = metrics
            if parameters is not None:
                self.recorder.final_parameters = fl.common.parameters_to_ndarrays(parameters)
            return parameters, metrics

        def aggregate_evaluate(self, server_round, results, failures):  # type: ignore[override]
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            while len(self.recorder.round_history) < server_round:
                self.recorder.round_history.append({"round": len(self.recorder.round_history) + 1})
            self.recorder.round_history[server_round - 1]["evaluate_loss"] = loss
            self.recorder.round_history[server_round - 1]["evaluate_metrics"] = metrics
            return loss, metrics

else:

    class TrackingFedAvg:
        """Placeholder used when Flower is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(FLOWER_IMPORT_ERROR_MESSAGE)


@dataclass(frozen=True)
class FedAvgStrategyFactory:
    """Default strategy factory for the stage-1 predictive baseline."""

    training_config: FederatedTrainingConfig

    def create(
        self,
        initial_parameters: list[np.ndarray],
        recorder: FederatedRunRecorder,
    ) -> Any:
        if fl is None:  # pragma: no cover - depends on optional deps
            raise ImportError(FLOWER_IMPORT_ERROR_MESSAGE)
        minimum_clients = min(self.training_config.min_available_clients, self.training_config.num_clients)
        return TrackingFedAvg(
            recorder=recorder,
            fraction_fit=self.training_config.fit_fraction,
            fraction_evaluate=self.training_config.evaluate_fraction,
            min_fit_clients=minimum_clients,
            min_evaluate_clients=minimum_clients,
            min_available_clients=minimum_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
            fit_metrics_aggregation_fn=_aggregate_scalar_metrics,
            evaluate_metrics_aggregation_fn=_aggregate_scalar_metrics,
        )


DEFAULT_STRATEGY_REGISTRY = StrategyRegistry(
    specs=[
        StrategySpec(
            key="fedavg",
            display_name="FedAvg",
            build_factory=lambda training_config: FedAvgStrategyFactory(training_config),
        )
    ]
)


def create_strategy_factory(
    strategy_name: str,
    *,
    training_config: FederatedTrainingConfig,
    registry: StrategyRegistry | None = None,
) -> StrategyFactory:
    """Build one configured strategy factory from the registry."""

    spec = (registry or DEFAULT_STRATEGY_REGISTRY).get(strategy_name)
    return spec.build_factory(training_config)


def _aggregate_scalar_metrics(
    metrics: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    numeric_rows: list[tuple[int, dict[str, float]]] = []
    for num_examples, row in metrics:
        numeric_row = {
            key: float(value)
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "client_id"
        }
        if numeric_row:
            numeric_rows.append((num_examples, numeric_row))
    return aggregate_weighted_metrics(numeric_rows) if numeric_rows else {}
