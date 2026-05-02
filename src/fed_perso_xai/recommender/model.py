"""Pairwise recommenders for explanation-candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_RECOMMENDER_TYPE = "svm_rank"
SUPPORTED_RECOMMENDER_TYPES = ("svm_rank", "pairwise_logistic")
_ARTIFACT_MODEL_TYPE_BY_RECOMMENDER = {
    "svm_rank": "svm_rank_recommender",
    "pairwise_logistic": "pairwise_logistic_recommender",
}
_RECOMMENDER_TYPE_BY_ARTIFACT_MODEL_TYPE = {
    artifact_model_type: recommender_type
    for recommender_type, artifact_model_type in _ARTIFACT_MODEL_TYPE_BY_RECOMMENDER.items()
}


def normalize_recommender_type(recommender_type: str) -> str:
    """Validate and normalize a user-facing recommender type key."""

    normalized = str(recommender_type).strip().lower()
    if normalized not in SUPPORTED_RECOMMENDER_TYPES:
        supported = ", ".join(SUPPORTED_RECOMMENDER_TYPES)
        raise ValueError(
            f"Unsupported recommender_type {recommender_type!r}. Supported values: {supported}."
        )
    return normalized


def recommender_artifact_model_type(recommender_type: str) -> str:
    """Return the persisted model_type string for one recommender backend."""

    return _ARTIFACT_MODEL_TYPE_BY_RECOMMENDER[normalize_recommender_type(recommender_type)]


def initialize_recommender_parameters(
    n_features: int,
    recommender_type: str = DEFAULT_RECOMMENDER_TYPE,
) -> list[np.ndarray]:
    """Return zero-initialized recommender parameters."""

    normalize_recommender_type(recommender_type)
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
    svm_c: float = 1.0
    svm_intercept_scaling: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.l2_regularization < 0:
            raise ValueError("l2_regularization must be >= 0.")
        if self.svm_c <= 0:
            raise ValueError("svm_c must be > 0.")
        if self.svm_intercept_scaling <= 0:
            raise ValueError("svm_intercept_scaling must be > 0.")


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

        return _score_candidates(
            self,
            candidates,
            feature_columns,
            variant_column=variant_column,
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
            svm_c=np.asarray([1.0], dtype=np.float64),
            svm_intercept_scaling=np.asarray([1.0], dtype=np.float64),
            model_type=np.asarray([recommender_artifact_model_type("pairwise_logistic")]),
        )
        return path


@dataclass
class SVMRankRecommender:
    """Linear squared-hinge ranker over pairwise candidate difference vectors.

    This tracks the older Perso-XAI ``LinearSVC`` objective more closely:
    squared hinge loss with a liblinear-style ``C`` parameter and intercept
    regularization through a synthetic-feature scaling term.
    """

    n_features: int
    learning_rate: float
    batch_size: int
    local_epochs: int
    svm_c: float = 1.0
    intercept_scaling: float = 1.0

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = np.zeros(1, dtype=np.float64)

    @classmethod
    def from_config(
        cls,
        *,
        n_features: int,
        config: PairwiseLogisticConfig,
    ) -> "SVMRankRecommender":
        return cls(
            n_features=n_features,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            local_epochs=config.epochs,
            svm_c=config.svm_c,
            intercept_scaling=config.svm_intercept_scaling,
        )

    def get_parameters(self) -> list[np.ndarray]:
        return [self.weights.copy(), self.bias.copy()]

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        if len(parameters) != 2:
            raise ValueError("SVMRankRecommender expects [weights, bias].")
        weights = np.asarray(parameters[0], dtype=np.float64).reshape(-1)
        if weights.shape[0] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} weights, received {weights.shape[0]}."
            )
        self.weights = weights.copy()
        self.bias = np.asarray(parameters[1], dtype=np.float64).reshape(1).copy()

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> float:
        """Train locally on a LinearSVC-style squared-hinge objective.

        The target objective is the liblinear primal form

            0.5 * ||w_aug||^2 + C * sum_i max(0, 1 - y_i f(x_i))^2

        where ``w_aug`` includes the synthetic intercept-feature weight. We
        still optimize it with mini-batch SGD so the model remains compatible
        with FedAvg parameter exchange.
        """

        X = _as_2d_float_array(X, n_features=self.n_features)
        y = _as_binary_labels(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        if X.shape[0] == 0:
            raise ValueError("Cannot train recommender on an empty dataset.")

        # Hinge-loss SVMs use signed class labels because the margin is defined
        # as y * f(x). Our pairwise dataset stores winners as binary labels
        # (1 => pair_1 preferred, 0 => pair_2 preferred), so we remap them to
        # +1 / -1 before computing margins and gradients.
        y_signed = np.where(y > 0.5, 1.0, -1.0)
        rng = np.random.default_rng(seed)
        sample_count = float(X.shape[0])
        batch_size = max(1, min(self.batch_size, X.shape[0]))
        for _ in range(self.local_epochs):
            indices = rng.permutation(X.shape[0])
            for start in range(0, X.shape[0], batch_size):
                batch_indices = indices[start : start + batch_size]
                X_batch = X[batch_indices]
                y_batch = y_signed[batch_indices]
                margins = y_batch * self.predict_pairwise_logits(X_batch)
                deficits = 1.0 - margins
                active = deficits > 0.0
                # We optimize a sample-count-normalized form of the liblinear
                # objective. This preserves the same minimizer as the summed
                # objective while keeping SGD updates well-scaled across clients.
                grad_w = self.weights / sample_count
                grad_b = self.bias[0] / ((self.intercept_scaling**2) * sample_count)
                if np.any(active):
                    signed_deficits = y_batch[active] * deficits[active]
                    batch_denominator = float(X_batch.shape[0])
                    grad_w -= 2.0 * self.svm_c * (
                        X_batch[active].T @ signed_deficits
                    ) / batch_denominator
                    grad_b -= 2.0 * self.svm_c * (
                        float(np.sum(signed_deficits)) / batch_denominator
                    )
                self.weights -= self.learning_rate * grad_w
                self.bias[0] -= self.learning_rate * grad_b
        return self.loss(X, y)

    def predict_pairwise_logits(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d_float_array(X, n_features=self.n_features)
        return X @ self.weights + self.bias[0]

    def predict_pairwise_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.clip(self.predict_pairwise_logits(X), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_pairwise(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        return (self.predict_pairwise_logits(X) >= threshold).astype(np.int64)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        X = _as_2d_float_array(X, n_features=self.n_features)
        y = _as_binary_labels(y)
        y_signed = np.where(y > 0.5, 1.0, -1.0)
        margins = y_signed * self.predict_pairwise_logits(X)
        squared_hinge = np.square(np.maximum(0.0, 1.0 - margins))
        regularization = 0.5 * (
            float(np.sum(self.weights**2))
            + float((self.bias[0] / self.intercept_scaling) ** 2)
        )
        return float(regularization + self.svm_c * float(np.sum(squared_hinge)))

    def score_candidate_matrix(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d_float_array(X, n_features=self.n_features)
        return X @ self.weights + self.bias[0]

    def score_candidates(
        self,
        candidates: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        variant_column: str = "method_variant",
    ) -> pd.Series:
        return _score_candidates(
            self,
            candidates,
            feature_columns,
            variant_column=variant_column,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights.astype(np.float64, copy=False),
            bias=self.bias.astype(np.float64, copy=False),
            n_features=np.asarray([self.n_features], dtype=np.int64),
            learning_rate=np.asarray([self.learning_rate], dtype=np.float64),
            batch_size=np.asarray([self.batch_size], dtype=np.int64),
            local_epochs=np.asarray([self.local_epochs], dtype=np.int64),
            svm_c=np.asarray([self.svm_c], dtype=np.float64),
            svm_intercept_scaling=np.asarray([self.intercept_scaling], dtype=np.float64),
            model_type=np.asarray([recommender_artifact_model_type("svm_rank")]),
        )
        return path


PairwiseRecommenderModel = PairwiseLogisticRecommender | SVMRankRecommender


def create_recommender(
    *,
    recommender_type: str,
    n_features: int,
    config: PairwiseLogisticConfig,
) -> PairwiseRecommenderModel:
    """Construct one supported recommender backend from shared hyperparameters."""

    normalized = normalize_recommender_type(recommender_type)
    if normalized == "svm_rank":
        return SVMRankRecommender.from_config(n_features=n_features, config=config)
    return PairwiseLogisticRecommender.from_config(n_features=n_features, config=config)


def _score_candidates(
    model: PairwiseLogisticRecommender | SVMRankRecommender,
    candidates: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    variant_column: str = "method_variant",
) -> pd.Series:
    if variant_column not in candidates.columns:
        raise ValueError(f"Missing candidate variant column {variant_column!r}.")
    X = candidates.loc[:, list(feature_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)
    scores = model.score_candidate_matrix(X.to_numpy(dtype=float))
    return pd.Series(
        scores,
        index=candidates[variant_column].astype(str),
        name="score",
    )


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

    with np.load(path, allow_pickle=False) as bundle:
        n_features = int(np.asarray(bundle["n_features"]).reshape(-1)[0])
        config = _config_from_bundle(bundle)
        model = PairwiseLogisticRecommender.from_config(n_features=n_features, config=config)
        model.set_parameters([bundle["weights"], bundle["bias"]])
    return model


def load_svm_rank_recommender(path: Path) -> SVMRankRecommender:
    """Load a persisted SVM rank recommender."""

    with np.load(path, allow_pickle=False) as bundle:
        n_features = int(np.asarray(bundle["n_features"]).reshape(-1)[0])
        config = _config_from_bundle(bundle)
        model = SVMRankRecommender.from_config(n_features=n_features, config=config)
        model.set_parameters([bundle["weights"], bundle["bias"]])
    return model


def load_recommender(path: Path) -> PairwiseRecommenderModel:
    """Load any persisted recommender artifact."""

    with np.load(path, allow_pickle=False) as bundle:
        raw_model_type = bundle.get(
            "model_type",
            np.asarray([recommender_artifact_model_type("pairwise_logistic")]),
        )
        artifact_model_type = str(np.asarray(raw_model_type).reshape(-1)[0])
    recommender_type = _RECOMMENDER_TYPE_BY_ARTIFACT_MODEL_TYPE.get(
        artifact_model_type,
        "pairwise_logistic",
    )
    if recommender_type == "svm_rank":
        return load_svm_rank_recommender(path)
    return load_pairwise_logistic_recommender(path)


def _config_from_bundle(bundle: np.lib.npyio.NpzFile) -> PairwiseLogisticConfig:
    return PairwiseLogisticConfig(
        epochs=int(np.asarray(bundle["local_epochs"]).reshape(-1)[0]),
        batch_size=int(np.asarray(bundle["batch_size"]).reshape(-1)[0]),
        learning_rate=float(np.asarray(bundle["learning_rate"]).reshape(-1)[0]),
        l2_regularization=float(
            np.asarray(bundle.get("l2_regularization", np.asarray([0.0]))).reshape(-1)[0]
        ),
        svm_c=float(np.asarray(bundle.get("svm_c", np.asarray([1.0]))).reshape(-1)[0]),
        svm_intercept_scaling=float(
            np.asarray(bundle.get("svm_intercept_scaling", np.asarray([1.0]))).reshape(-1)[0]
        ),
    )
