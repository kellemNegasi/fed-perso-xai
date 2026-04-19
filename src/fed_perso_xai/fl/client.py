"""Flower client adapters for the stage-1 federated baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

from fed_perso_xai.evaluation.metrics import compute_classification_metrics
from fed_perso_xai.models.logistic_regression import LogisticRegressionModel
from fed_perso_xai.utils.config import LogisticRegressionConfig


@dataclass(frozen=True)
class ClientData:
    """Local client arrays used in training and evaluation."""

    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    row_ids_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    row_ids_test: np.ndarray


if fl is not None:

    class FederatedLogisticRegressionClient(fl.client.NumPyClient):
        """Flower NumPy client backed by the explicit NumPy logistic regression model."""

        def __init__(
            self,
            data: ClientData,
            model_config: LogisticRegressionConfig,
            seed: int,
        ) -> None:
            self.data = data
            self.seed = seed
            self.model = LogisticRegressionModel(
                n_features=data.X_train.shape[1],
                learning_rate=model_config.learning_rate,
                batch_size=model_config.batch_size,
                local_epochs=model_config.epochs,
                l2_regularization=model_config.l2_regularization,
            )

        def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
            return self.model.get_parameters()

        def fit(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
        ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
            self.model.set_parameters(parameters)
            train_loss = self.model.fit(
                self.data.X_train,
                self.data.y_train,
                seed=self.seed + self.data.client_id,
            )
            metrics: dict[str, Any] = {
                "train_loss": float(train_loss),
                "client_id": str(self.data.client_id),
            }
            return self.model.get_parameters(), int(self.data.y_train.shape[0]), metrics

        def evaluate(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
        ) -> tuple[float, int, dict[str, Any]]:
            self.model.set_parameters(parameters)
            loss = self.model.loss(self.data.X_test, self.data.y_test)
            probabilities = self.model.predict_proba(self.data.X_test)
            metrics = compute_classification_metrics(self.data.y_test, probabilities, loss)
            metrics["client_id"] = str(self.data.client_id)
            return float(loss), int(self.data.y_test.shape[0]), metrics

else:

    class FederatedLogisticRegressionClient:
        """Placeholder used when Flower is not installed."""

        def __init__(
            self,
            data: ClientData,
            model_config: LogisticRegressionConfig,
            seed: int,
        ) -> None:
            raise ImportError(FLOWER_IMPORT_ERROR_MESSAGE)
