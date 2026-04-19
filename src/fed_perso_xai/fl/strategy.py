"""Flower strategy helpers with future extension seams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import flwr as fl
import numpy as np

from fed_perso_xai.evaluation.metrics import aggregate_weighted_metrics
from fed_perso_xai.utils.config import FederatedTrainingConfig


class StrategyFactory(Protocol):
    """Protocol for pluggable aggregation strategies in later stages."""

    def create(self, initial_parameters: list[np.ndarray]) -> fl.server.strategy.Strategy:
        """Build a Flower strategy."""


@dataclass(frozen=True)
class FedAvgStrategyFactory:
    """Default strategy factory for the stage-1 predictive baseline."""

    training_config: FederatedTrainingConfig

    def create(self, initial_parameters: list[np.ndarray]) -> fl.server.strategy.Strategy:
        return fl.server.strategy.FedAvg(
            fraction_fit=self.training_config.fit_fraction,
            fraction_evaluate=self.training_config.evaluate_fraction,
            min_fit_clients=min(
                self.training_config.min_available_clients,
                self.training_config.num_clients,
            ),
            min_evaluate_clients=min(
                self.training_config.min_available_clients,
                self.training_config.num_clients,
            ),
            min_available_clients=min(
                self.training_config.min_available_clients,
                self.training_config.num_clients,
            ),
            initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
            fit_metrics_aggregation_fn=_aggregate_scalar_metrics,
            evaluate_metrics_aggregation_fn=_aggregate_scalar_metrics,
        )


def _aggregate_scalar_metrics(
    metrics: list[tuple[int, dict[str, fl.common.Scalar]]],
) -> dict[str, fl.common.Scalar]:
    """Aggregate Flower metric payloads using weighted averages where possible."""

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
