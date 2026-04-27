"""Flower client adapters for the federated baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import logging
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
from fed_perso_xai.models import create_model
from fed_perso_xai.recommender import PairwiseLogisticConfig, PairwiseLogisticRecommender

LOGGER = logging.getLogger(__name__)


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

    def get_split(self, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return one local split in a consistent order."""

        normalized = split_name.strip().lower()
        if normalized in {"train", "local_train", "client_local_train"}:
            return self.X_train, self.y_train, self.row_ids_train
        if normalized in {"test", "local_test", "client_local_test"}:
            return self.X_test, self.y_test, self.row_ids_test
        raise ValueError(f"Unsupported client split '{split_name}'. Expected 'train' or 'test'.")


@dataclass(frozen=True)
class RecommenderClientData:
    """Local pairwise arrays and held-out context for recommender FL."""

    client_id: int
    client_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray


@dataclass(frozen=True)
class SharedParameterPayload:
    """Subset of model parameters that participates in server aggregation.

    The current baseline has no personalized server-excluded tensors yet, so all model
    parameters are currently treated as shared/global. The helper is kept
    explicit so later iterations can leave local tensors on the client.
    """

    shared_parameters: list[np.ndarray]
    shared_parameter_indices: tuple[int, ...]
    total_parameter_count: int


def extract_shared_parameter_payload(parameters: list[np.ndarray]) -> SharedParameterPayload:
    """Return the model tensors that should be aggregated by the server."""

    normalized = [np.asarray(parameter, dtype=np.float64).copy() for parameter in parameters]
    return SharedParameterPayload(
        shared_parameters=normalized,
        shared_parameter_indices=tuple(range(len(normalized))),
        total_parameter_count=len(normalized),
    )


def apply_shared_parameter_payload(
    current_parameters: list[np.ndarray],
    shared_parameters: list[np.ndarray],
    shared_parameter_indices: tuple[int, ...] | None = None,
) -> list[np.ndarray]:
    """Merge aggregated shared tensors back into a full local model state."""

    merged = [np.asarray(parameter, dtype=np.float64).copy() for parameter in current_parameters]
    indices = shared_parameter_indices or tuple(range(len(shared_parameters)))
    if len(shared_parameters) != len(indices):
        raise ValueError("shared_parameters and shared_parameter_indices must align.")
    if len(indices) > len(merged):
        raise ValueError("shared_parameter_indices exceeds the local parameter count.")

    for index, parameter in zip(indices, shared_parameters, strict=True):
        merged[index] = np.asarray(parameter, dtype=np.float64).copy()
    return merged


if fl is not None:

    class FederatedLogisticRegressionClient(fl.client.NumPyClient):
        """Flower NumPy client backed by the explicit NumPy logistic regression model."""

        def __init__(
            self,
            data: ClientData,
            model_name: str,
            model_config: Any,
            seed: int,
            prediction_threshold: float = 0.5,
        ) -> None:
            self.data = data
            self.seed = seed
            self.prediction_threshold = float(prediction_threshold)
            self.model = create_model(
                model_name,
                n_features=data.X_train.shape[1],
                config=model_config,
            )

        def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
            return extract_shared_parameter_payload(self.model.get_parameters()).shared_parameters

        def fit(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
        ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
            merged_parameters = apply_shared_parameter_payload(
                self.model.get_parameters(),
                parameters,
            )
            self.model.set_parameters(merged_parameters)
            train_loss = self.model.fit(
                self.data.X_train,
                self.data.y_train,
                seed=self.seed + self.data.client_id,
            )
            shared_payload = extract_shared_parameter_payload(self.model.get_parameters())
            metrics: dict[str, Any] = {
                "train_loss": float(train_loss),
                "client_id": str(self.data.client_id),
                # The current baseline aggregates the full predictive model. Future versions
                # may introduce explicit shared/local parameter splits.
                "aggregation_scope": "full_model",
                "shared_parameter_count": int(len(shared_payload.shared_parameters)),
                "shared_parameter_indices": ",".join(
                    str(index) for index in shared_payload.shared_parameter_indices
                ),
            }
            return (
                shared_payload.shared_parameters,
                int(self.data.y_train.shape[0]),
                metrics,
            )

        def evaluate(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
        ) -> tuple[float, int, dict[str, Any]]:
            merged_parameters = apply_shared_parameter_payload(
                self.model.get_parameters(),
                parameters,
            )
            self.model.set_parameters(merged_parameters)
            loss = self.model.loss(self.data.X_test, self.data.y_test)
            probabilities = self.model.predict_proba(self.data.X_test)
            metrics = compute_classification_metrics(
                self.data.y_test,
                probabilities,
                loss,
                threshold=self.prediction_threshold,
            )
            metrics["client_id"] = str(self.data.client_id)
            return float(loss), int(self.data.y_test.shape[0]), metrics


    class FederatedPairwiseRecommenderClient(fl.client.NumPyClient):
        """Flower NumPy client backed by the pairwise logistic recommender."""

        def __init__(
            self,
            data: RecommenderClientData,
            model_config: PairwiseLogisticConfig,
            seed: int,
        ) -> None:
            self.data = data
            self.seed = int(seed)
            self.model = PairwiseLogisticRecommender.from_config(
                n_features=data.X_train.shape[1],
                config=model_config,
            )

        def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
            return extract_shared_parameter_payload(self.model.get_parameters()).shared_parameters

        def fit(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
        ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
            LOGGER.info(
                "Recommender fit start client=%s train_pairs=%s",
                self.data.client_name,
                int(self.data.y_train.shape[0]),
            )
            merged_parameters = apply_shared_parameter_payload(
                self.model.get_parameters(),
                parameters,
            )
            self.model.set_parameters(merged_parameters)
            train_loss = self.model.fit(
                self.data.X_train,
                self.data.y_train,
                seed=self.seed + self.data.client_id,
            )
            shared_payload = extract_shared_parameter_payload(self.model.get_parameters())
            metrics: dict[str, Any] = {
                "train_loss": float(train_loss),
                "client_id": self.data.client_name,
                "aggregation_scope": "full_recommender",
                "shared_parameter_count": int(len(shared_payload.shared_parameters)),
                "shared_parameter_indices": ",".join(
                    str(index) for index in shared_payload.shared_parameter_indices
                ),
            }
            LOGGER.info(
                "Recommender fit complete client=%s train_pairs=%s train_loss=%.6f",
                self.data.client_name,
                int(self.data.y_train.shape[0]),
                float(train_loss),
            )
            return (
                shared_payload.shared_parameters,
                int(self.data.y_train.shape[0]),
                metrics,
            )

        def evaluate(
            self,
            parameters: list[np.ndarray],
            config: dict[str, Any],
        ) -> tuple[float, int, dict[str, Any]]:
            eval_pairs = int(self.data.y_eval.shape[0])
            LOGGER.info(
                "Recommender eval start client=%s eval_pairs=%s",
                self.data.client_name,
                eval_pairs,
            )
            if eval_pairs == 0:
                LOGGER.info(
                    "Recommender eval skipped client=%s eval_pairs=0",
                    self.data.client_name,
                )
                return 0.0, 0, {"client_id": self.data.client_name}
            merged_parameters = apply_shared_parameter_payload(
                self.model.get_parameters(),
                parameters,
            )
            self.model.set_parameters(merged_parameters)
            loss = self.model.loss(self.data.X_eval, self.data.y_eval)
            predictions = self.model.predict_pairwise(self.data.X_eval)
            accuracy = float(np.mean(predictions == self.data.y_eval))
            metrics: dict[str, Any] = {
                "client_id": self.data.client_name,
                "pairwise_accuracy": accuracy,
            }
            LOGGER.info(
                "Recommender eval complete client=%s eval_pairs=%s eval_loss=%.6f pairwise_accuracy=%.6f",
                self.data.client_name,
                eval_pairs,
                float(loss),
                accuracy,
            )
            return float(loss), eval_pairs, metrics


else:

    class FederatedLogisticRegressionClient:
        """Placeholder used when Flower is not installed."""

        def __init__(
            self,
            data: ClientData,
            model_name: str,
            model_config: Any,
            seed: int,
        ) -> None:
            raise ImportError(FLOWER_IMPORT_ERROR_MESSAGE)

    class FederatedPairwiseRecommenderClient:
        """Placeholder used when Flower is not installed."""

        def __init__(
            self,
            data: RecommenderClientData,
            model_config: PairwiseLogisticConfig,
            seed: int,
        ) -> None:
            raise ImportError(FLOWER_IMPORT_ERROR_MESSAGE)

