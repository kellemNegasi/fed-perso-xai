"""NumPy pairwise logistic recommender for explanation-candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def initialize_recommender_parameters(n_features: int) -> list[np.ndarray]:
    """Return zero-initialized pairwise logistic recommender parameters."""

    return [
        np.zeros(n_features, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    ]


@dataclass
class PairwiseLogisticConfig:
    """Hyperparameters for local recommender training."""

    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.05
    l2_regularization: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.l2_regularization < 0:
            raise ValueError("l2_regularization must be >= 0.")


@dataclass
class PairwiseLogisticRecommender:
    """Binary logistic model over pairwise candidate difference vectors.

    Training examples are expected to be `features(pair_1) - features(pair_2)`.
    Labels are binary:
    - `1`: pair_1 is preferred over pair_2
    - `0`: pair_2 is preferred over pair_1

    The learned weight vector can score individual candidates with
    `utility(candidate) = features(candidate) @ weights`. The pairwise bias is
    used for pair classification but intentionally omitted from individual
    candidate ranking because it is constant across candidates.
    """

    n_features: int
    learning_rate: float
    batch_size: int
    local_epochs: int
    l2_regularization: float = 0.0

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = np.zeros(1, dtype=np.float64)

    @classmethod
    def from_config(
        cls,
        *,
        n_features: int,
        config: PairwiseLogisticConfig,
    ) -> "PairwiseLogisticRecommender":
        """Build a recommender from a typed config object."""

        return cls(
            n_features=n_features,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            local_epochs=config.epochs,
            l2_regularization=config.l2_regularization,
        )

    def get_parameters(self) -> list[np.ndarray]:
        """Return model parameters in FedAvg-compatible order."""

        return [self.weights.copy(), self.bias.copy()]

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        """Load model parameters in FedAvg-compatible order."""

        if len(parameters) != 2:
            raise ValueError("PairwiseLogisticRecommender expects [weights, bias].")
        weights = np.asarray(parameters[0], dtype=np.float64).reshape(-1)
        if weights.shape[0] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} weights, received {weights.shape[0]}."
            )
        self.weights = weights.copy()
        self.bias = np.asarray(parameters[1], dtype=np.float64).reshape(1).copy()

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> float:
        """Train locally with mini-batch SGD and return final binary cross-entropy."""

        X = _as_2d_float_array(X, n_features=self.n_features)
        y = _as_binary_labels(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        if X.shape[0] == 0:
            raise ValueError("Cannot train recommender on an empty dataset.")

        rng = np.random.default_rng(seed)
        batch_size = max(1, min(self.batch_size, X.shape[0]))
        for _ in range(self.local_epochs):
            indices = rng.permutation(X.shape[0])
            for start in range(0, X.shape[0], batch_size):
                batch_indices = indices[start : start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                probabilities = self.predict_pairwise_proba(X_batch)
                errors = probabilities - y_batch
                grad_w = (X_batch.T @ errors) / X_batch.shape[0]
                grad_w += self.l2_regularization * self.weights
                grad_b = float(np.mean(errors))
                self.weights -= self.learning_rate * grad_w
                self.bias[0] -= self.learning_rate * grad_b
        return self.loss(X, y)

    def predict_pairwise_logits(self, X: np.ndarray) -> np.ndarray:
        """Return logits for pairwise difference vectors."""

        X = _as_2d_float_array(X, n_features=self.n_features)
        return X @ self.weights + self.bias[0]

    def predict_pairwise_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(pair_1 preferred over pair_2) for difference vectors."""

        logits = np.clip(self.predict_pairwise_logits(X), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_pairwise(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary pairwise predictions."""

        return (self.predict_pairwise_proba(X) >= threshold).astype(np.int64)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return binary cross-entropy plus L2 regularization."""

        X = _as_2d_float_array(X, n_features=self.n_features)
        y = _as_binary_labels(y)
        probabilities = np.clip(self.predict_pairwise_proba(X), 1e-8, 1.0 - 1e-8)
        data_loss = -np.mean(
            y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities)
        )
        regularization = 0.5 * self.l2_regularization * float(np.sum(self.weights**2))
        return float(data_loss + regularization)

    def score_candidate_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return utility scores for individual candidate feature rows."""

        X = _as_2d_float_array(X, n_features=self.n_features)
        return X @ self.weights

    def score_candidates(
        self,
        candidates: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        variant_column: str = "method_variant",
    ) -> pd.Series:
        """Return utility scores indexed by candidate variant."""

        if variant_column not in candidates.columns:
            raise ValueError(f"Missing candidate variant column {variant_column!r}.")
        X = candidates.loc[:, list(feature_columns)].apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0.0)
        scores = self.score_candidate_matrix(X.to_numpy(dtype=float))
        return pd.Series(
            scores,
            index=candidates[variant_column].astype(str),
            name="score",
        )

    def save(self, path: Path) -> Path:
        """Persist trained recommender parameters."""

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
            model_type=np.asarray(["pairwise_logistic_recommender"]),
        )
        return path


def _as_2d_float_array(X: np.ndarray, *, n_features: int) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if arr.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, received {arr.shape[1]}.")
    return arr


def _as_binary_labels(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if not np.isin(arr, [0.0, 1.0]).all():
        raise ValueError("Pairwise labels must be binary values in {0, 1}.")
    return arr

def load_pairwise_logistic_recommender(path: Path) -> PairwiseLogisticRecommender:
    """Load a persisted pairwise logistic recommender."""

    bundle = np.load(path, allow_pickle=False)
    n_features = int(np.asarray(bundle["n_features"]).reshape(-1)[0])
    config = PairwiseLogisticConfig(
        epochs=int(np.asarray(bundle["local_epochs"]).reshape(-1)[0]),
        batch_size=int(np.asarray(bundle["batch_size"]).reshape(-1)[0]),
        learning_rate=float(np.asarray(bundle["learning_rate"]).reshape(-1)[0]),
        l2_regularization=float(np.asarray(bundle["l2_regularization"]).reshape(-1)[0]),
    )
    model = PairwiseLogisticRecommender.from_config(n_features=n_features, config=config)
    model.set_parameters([bundle["weights"], bundle["bias"]])
    return model

