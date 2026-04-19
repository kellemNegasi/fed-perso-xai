"""Explicit NumPy logistic regression for centralized and federated training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def initialize_parameters(n_features: int) -> list[np.ndarray]:
    """Return zero-initialized logistic regression parameters."""

    return [
        np.zeros(n_features, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    ]


@dataclass
class LogisticRegressionModel:
    """A minimal binary logistic regression trained with mini-batch SGD."""

    n_features: int
    learning_rate: float
    batch_size: int
    local_epochs: int
    l2_regularization: float = 0.0

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = np.zeros(1, dtype=np.float64)

    def get_parameters(self) -> list[np.ndarray]:
        """Return the current parameters."""

        return [self.weights.copy(), self.bias.copy()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load parameters received from the server."""

        self.weights = np.asarray(parameters[0], dtype=np.float64).copy()
        self.bias = np.asarray(parameters[1], dtype=np.float64).reshape(1).copy()

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> float:
        """Train locally and return the final training loss."""

        rng = np.random.default_rng(seed)
        n_samples = X.shape[0]
        batch_size = max(1, min(self.batch_size, n_samples))
        for _ in range(self.local_epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                predictions = self.predict_proba(X_batch)
                errors = predictions - y_batch
                grad_w = (X_batch.T @ errors) / X_batch.shape[0]
                grad_w += self.l2_regularization * self.weights
                grad_b = np.mean(errors)
                self.weights -= self.learning_rate * grad_w
                self.bias[0] -= self.learning_rate * grad_b
        return self.loss(X, y)

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Return raw logits."""

        return X @ self.weights + self.bias[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities."""

        logits = np.clip(self.predict_logits(X), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions."""

        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return binary cross-entropy loss."""

        probabilities = np.clip(self.predict_proba(X), 1e-8, 1.0 - 1e-8)
        data_loss = -np.mean(
            y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities)
        )
        regularization = 0.5 * self.l2_regularization * np.sum(self.weights**2)
        return float(data_loss + regularization)

    def save(self, path: Path) -> Path:
        """Persist the trained model parameters."""

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights.astype(np.float64, copy=False),
            bias=self.bias.astype(np.float64, copy=False),
            n_features=np.asarray([self.n_features], dtype=np.int64),
            learning_rate=np.asarray([self.learning_rate], dtype=np.float64),
            batch_size=np.asarray([self.batch_size], dtype=np.int64),
            local_epochs=np.asarray([self.local_epochs], dtype=np.int64),
            l2_regularization=np.asarray([self.l2_regularization], dtype=np.float64),
        )
        return path
