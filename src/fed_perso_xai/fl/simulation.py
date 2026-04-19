"""Flower training runner with a sequential fallback when Ray is unavailable."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
from flwr.common import Code, EvaluateRes, FitRes, Status
from flwr.server.client_proxy import ClientProxy

from fed_perso_xai.evaluation.metrics import (
    aggregate_weighted_metrics,
    compute_pooled_classification_metrics,
    sweep_classification_thresholds,
)
from fed_perso_xai.fl.client import ClientData, FederatedLogisticRegressionClient
from fed_perso_xai.fl.strategy import FedAvgStrategyFactory, StrategyFactory
from fed_perso_xai.models.logistic_regression import initialize_parameters
from fed_perso_xai.utils.config import FederatedTrainingConfig


@dataclass(frozen=True)
class SimulationArtifacts:
    """Training outputs for one federated run."""

    result_dir: Path
    metrics_path: Path
    model_path: Path


class InMemoryClientProxy(ClientProxy):
    """Minimal proxy used by the sequential fallback to satisfy Flower types."""

    def __init__(self, cid: str) -> None:
        super().__init__(cid=cid)

    def get_properties(self, ins, timeout, group_id):  # type: ignore[override]
        raise NotImplementedError

    def get_parameters(self, ins, timeout, group_id):  # type: ignore[override]
        raise NotImplementedError

    def fit(self, ins, timeout, group_id):  # type: ignore[override]
        raise NotImplementedError

    def evaluate(self, ins, timeout, group_id):  # type: ignore[override]
        raise NotImplementedError

    def reconnect(self, ins, timeout, group_id):  # type: ignore[override]
        raise NotImplementedError


def run_federated_training(
    *,
    client_datasets: list[ClientData],
    config: FederatedTrainingConfig,
    result_dir: Path,
    strategy_factory: StrategyFactory | None = None,
) -> tuple[SimulationArtifacts, dict[str, Any]]:
    """Run federated training and persist final metrics and parameters."""

    n_features = client_datasets[0].X_train.shape[1]
    initial_parameters = initialize_parameters(n_features)
    strategy = (strategy_factory or FedAvgStrategyFactory(config)).create(initial_parameters)
    clients = [
        FederatedLogisticRegressionClient(
            data=dataset,
            n_features=n_features,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            local_epochs=config.local_epochs,
            l2_regularization=config.l2_regularization,
            seed=config.seed,
        )
        for dataset in client_datasets
    ]
    summary = _run_sequential_fallback(
        clients=clients,
        config=config,
        strategy=strategy,
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / "metrics_summary.json"
    model_path = result_dir / "model_parameters.npz"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        model_path,
        weights=np.asarray(summary["final_parameters"][0], dtype=np.float64),
        bias=np.asarray(summary["final_parameters"][1], dtype=np.float64),
    )
    return SimulationArtifacts(result_dir=result_dir, metrics_path=metrics_path, model_path=model_path), summary


def _run_sequential_fallback(
    *,
    clients: list[FederatedLogisticRegressionClient],
    config: FederatedTrainingConfig,
    strategy: fl.server.strategy.Strategy,
) -> dict[str, Any]:
    """Use Flower strategy objects with in-process clients when Ray is unavailable."""

    rng = np.random.default_rng(config.seed)
    parameters = clients[0].get_parameters({})
    round_history: list[dict[str, Any]] = []
    proxies = [InMemoryClientProxy(cid=str(index)) for index, _ in enumerate(clients)]

    for server_round in range(1, config.rounds + 1):
        fit_sample_size = _sample_size(
            total_clients=len(clients),
            fraction=config.fit_fraction,
            minimum=min(config.min_available_clients, len(clients)),
        )
        fit_indices = rng.choice(len(clients), size=fit_sample_size, replace=False)
        fit_results: list[tuple[ClientProxy, FitRes]] = []
        for client_index in fit_indices:
            updated_parameters, num_examples, metrics = clients[client_index].fit(parameters, {})
            fit_results.append(
                (
                    proxies[client_index],
                    FitRes(
                        status=Status(code=Code.OK, message="ok"),
                        parameters=fl.common.ndarrays_to_parameters(updated_parameters),
                        num_examples=num_examples,
                        metrics=metrics,
                    ),
                )
            )

        aggregated_parameters, fit_metrics = strategy.aggregate_fit(server_round, fit_results, [])
        if aggregated_parameters is None:
            raise RuntimeError("Flower strategy returned no aggregated parameters.")
        parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)

        evaluate_sample_size = _sample_size(
            total_clients=len(clients),
            fraction=config.evaluate_fraction,
            minimum=min(config.min_available_clients, len(clients)),
        )
        evaluate_indices = rng.choice(len(clients), size=evaluate_sample_size, replace=False)
        evaluate_results: list[tuple[ClientProxy, EvaluateRes]] = []
        for client_index in evaluate_indices:
            loss, num_examples, metrics = clients[client_index].evaluate(parameters, {})
            evaluate_results.append(
                (
                    proxies[client_index],
                    EvaluateRes(
                        status=Status(code=Code.OK, message="ok"),
                        loss=loss,
                        num_examples=num_examples,
                        metrics=metrics,
                    ),
                )
            )

        aggregated_loss, evaluate_metrics = strategy.aggregate_evaluate(
            server_round,
            evaluate_results,
            [],
        )
        round_history.append(
            {
                "round": server_round,
                "fit_metrics": fit_metrics,
                "evaluate_loss": aggregated_loss,
                "evaluate_metrics": evaluate_metrics,
            }
        )

    per_client: list[dict[str, Any]] = []
    aggregated_inputs: list[tuple[int, dict[str, float]]] = []
    pooled_probabilities: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    for client in clients:
        loss, num_examples, metrics = client.evaluate(parameters, {})
        client.model.set_parameters(parameters)
        probabilities = client.model.predict_proba(client.data.X_test)
        row = {
            "client_id": client.data.client_id,
            "num_examples": num_examples,
            **{key: float(value) for key, value in metrics.items() if key != "client_id"},
        }
        per_client.append(row)
        aggregated_inputs.append(
            (
                num_examples,
                {key: float(value) for key, value in metrics.items() if key != "client_id"},
            )
        )
        pooled_probabilities.append(probabilities)
        pooled_labels.append(client.data.y_test)

    aggregated_metrics = aggregate_weighted_metrics(aggregated_inputs)
    pooled_y_true = np.concatenate(pooled_labels, axis=0)
    pooled_y_prob = np.concatenate(pooled_probabilities, axis=0)
    pooled_loss = float(
        np.average(
            [row["loss"] for row in per_client],
            weights=[row["num_examples"] for row in per_client],
        )
    )
    pooled_metrics = compute_pooled_classification_metrics(
        y_true=pooled_y_true,
        y_prob=pooled_y_prob,
        loss=pooled_loss,
    )
    threshold_sweep = sweep_classification_thresholds(
        y_true=pooled_y_true,
        y_prob=pooled_y_prob,
    )
    return {
        "dataset_name": config.dataset_name,
        "simulation_backend": "sequential_flower_fallback",
        "config": config.to_dict(),
        "round_history": round_history,
        "per_client": per_client,
        "aggregated_weighted": aggregated_metrics,
        "aggregated_pooled": pooled_metrics,
        "threshold_analysis": threshold_sweep,
        "final_parameters": [parameters[0].tolist(), parameters[1].tolist()],
    }


def _sample_size(total_clients: int, fraction: float, minimum: int) -> int:
    sample_size = int(np.ceil(total_clients * fraction))
    return max(1, min(total_clients, max(minimum, sample_size)))
