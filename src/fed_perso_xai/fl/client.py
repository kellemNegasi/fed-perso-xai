"""Flower NumPy client for the stage-1 baseline."""

from __future__ import annotations

from dataclasses import dataclass

import flwr as fl
import numpy as np

from fed_perso_xai.evaluation.metrics import compute_classification_metrics
from fed_perso_xai.models.logistic_regression import LogisticRegressionModel


@dataclass(frozen=True)
class ClientData:
    """Local client arrays used in training and evaluation."""

    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


class FederatedLogisticRegressionClient(fl.client.NumPyClient):
    """Flower client backed by the explicit NumPy logistic regression model."""

    def __init__(
        self,
        data: ClientData,
        n_features: int,
        learning_rate: float,
        batch_size: int,
        local_epochs: int,
        l2_regularization: float,
        seed: int,
    ) -> None:
        self.data = data
        self.seed = seed
        self.model = LogisticRegressionModel(
            n_features=n_features,
            learning_rate=learning_rate,
            batch_size=batch_size,
            local_epochs=local_epochs,
            l2_regularization=l2_regularization,
        )

    def get_parameters(self, config: dict[str, fl.common.Scalar]) -> list[np.ndarray]:
        return self.model.get_parameters()

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, fl.common.Scalar],
    ) -> tuple[list[np.ndarray], int, dict[str, fl.common.Scalar]]:
        self.model.set_parameters(parameters)
        train_loss = self.model.fit(
            self.data.X_train,
            self.data.y_train,
            seed=self.seed + self.data.client_id,
        )
        metrics: dict[str, fl.common.Scalar] = {
            "train_loss": float(train_loss),
            "client_id": str(self.data.client_id),
        }
        return self.model.get_parameters(), int(self.data.y_train.shape[0]), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, fl.common.Scalar],
    ) -> tuple[float, int, dict[str, fl.common.Scalar]]:
        self.model.set_parameters(parameters)
        loss = self.model.loss(self.data.X_test, self.data.y_test)
        probabilities = self.model.predict_proba(self.data.X_test)
        metrics = compute_classification_metrics(self.data.y_test, probabilities, loss)
        metrics["client_id"] = str(self.data.client_id)
        return float(loss), int(self.data.y_test.shape[0]), metrics
