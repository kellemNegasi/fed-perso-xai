"""Flower runtime integration for stage-1 federated experiments."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FLOWER_IMPORT_ERROR_MESSAGE = (
    "Flower support is not installed. Install the optional federated extras with "
    "`pip install -e .[fl]` for debug runtime support or `pip install -e .[ray]` "
    "for Ray-backed simulation."
)

try:
    import flwr as fl
    from flwr.clientapp import ClientApp
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
except ImportError:  # pragma: no cover - exercised via optional dependency paths
    fl = None  # type: ignore[assignment]
    ClientApp = None  # type: ignore[assignment]
    ServerApp = None  # type: ignore[assignment]
    ServerAppComponents = None  # type: ignore[assignment]
    ServerConfig = None  # type: ignore[assignment]

from fed_perso_xai.evaluation.metrics import (
    aggregate_weighted_metrics,
    compute_pooled_classification_metrics,
    summarize_class_balance,
    summarize_probability_distribution,
    sweep_classification_thresholds,
)
from fed_perso_xai.evaluation.predictions import (
    build_prediction_artifact,
    save_prediction_artifact,
)
from fed_perso_xai.fl.client import ClientData, FederatedLogisticRegressionClient
from fed_perso_xai.fl.strategy import (
    FedAvgStrategyFactory,
    FederatedRunRecorder,
    StrategyFactory,
)
from fed_perso_xai.models.logistic_regression import LogisticRegressionModel, initialize_parameters
from fed_perso_xai.utils.config import FederatedTrainingConfig


BACKEND_ALIASES = {
    "auto": "auto",
    "ray": "ray",
    "debug-sequential": "debug-sequential",
    "sequential_fallback": "debug-sequential",
}


@dataclass(frozen=True)
class FlowerRuntimePlan:
    """Resolved runtime plan for a federated experiment."""

    requested_backend: str
    resolved_backend: str
    ray_available: bool
    flwr_version: str
    is_debug_runtime: bool
    warnings: list[str]


@dataclass(frozen=True)
class SimulationArtifacts:
    """Training outputs for one federated run."""

    result_dir: Path
    metrics_path: Path
    model_path: Path
    predictions_path: Path
    runtime_path: Path


def plan_flower_runtime(config: FederatedTrainingConfig) -> FlowerRuntimePlan:
    """Resolve the Flower runtime before federated training starts."""

    require_flower_support()
    requested_backend = _canonicalize_backend(config.simulation_backend)
    ray_available = importlib.util.find_spec("ray") is not None
    warnings: list[str] = []

    if requested_backend == "auto":
        if not ray_available:
            raise RuntimeError(
                "Flower simulation requires Ray, but Ray is not installed. "
                "Install `fed-perso-xai[ray]` for the primary simulation path or "
                "rerun with `--simulation-backend debug-sequential` for the debug-only "
                "development fallback."
            )
        resolved_backend = "ray"
    elif requested_backend == "ray":
        if not ray_available:
            raise RuntimeError(
                "The requested Flower runtime backend is `ray`, but Ray is not installed. "
                "Install `fed-perso-xai[ray]` or use `--simulation-backend debug-sequential` "
                "for the explicit debug fallback."
            )
        resolved_backend = "ray"
    else:
        resolved_backend = "debug-sequential"
        warnings.append(
            "Using the debug sequential runtime. This is not a real Flower simulation "
            "and should only be used for local development or CI fallback checks."
        )

    return FlowerRuntimePlan(
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        ray_available=ray_available,
        flwr_version=importlib.metadata.version("flwr"),
        is_debug_runtime=resolved_backend == "debug-sequential",
        warnings=warnings,
    )


def run_federated_training(
    *,
    client_datasets: list[ClientData],
    config: FederatedTrainingConfig,
    result_dir: Path,
    run_id: str,
    strategy_factory: StrategyFactory | None = None,
) -> tuple[SimulationArtifacts, dict[str, Any]]:
    """Run federated training and persist final metrics and parameters."""

    require_flower_support()
    if not client_datasets:
        raise ValueError("client_datasets must not be empty.")

    result_dir.mkdir(parents=True, exist_ok=True)
    runtime_plan = plan_flower_runtime(config)
    initial_parameters = initialize_parameters(client_datasets[0].X_train.shape[1])
    recorder = FederatedRunRecorder(backend=runtime_plan.resolved_backend)
    factory = strategy_factory or FedAvgStrategyFactory(config)
    final_parameters, actual_backend, runtime_warnings = _execute_runtime(
        client_datasets=client_datasets,
        config=config,
        runtime_plan=runtime_plan,
        recorder=recorder,
        strategy_factory=factory,
        initial_parameters=initial_parameters,
    )

    summary, predictions_path = _evaluate_and_persist_results(
        client_datasets=client_datasets,
        config=config,
        result_dir=result_dir,
        recorder=recorder,
        final_parameters=final_parameters,
        run_id=run_id,
        actual_backend=actual_backend,
    )
    runtime_report = {
        "requested_backend": runtime_plan.requested_backend,
        "planned_backend": runtime_plan.resolved_backend,
        "actual_backend": actual_backend,
        "ray_available": runtime_plan.ray_available,
        "flwr_version": runtime_plan.flwr_version,
        "is_debug_runtime": actual_backend == "debug-sequential",
        "warnings": runtime_plan.warnings + runtime_warnings,
        "rounds_completed": len(recorder.round_history),
    }
    runtime_path = result_dir / "runtime_report.json"
    runtime_path.write_text(json.dumps(runtime_report, indent=2), encoding="utf-8")

    metrics_path = result_dir / "metrics_summary.json"
    model_path = result_dir / "model_parameters.npz"
    summary["runtime"] = runtime_report
    summary["simulation_backend"] = actual_backend
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        model_path,
        weights=np.asarray(final_parameters[0], dtype=np.float64),
        bias=np.asarray(final_parameters[1], dtype=np.float64),
    )
    return (
        SimulationArtifacts(
            result_dir=result_dir,
            metrics_path=metrics_path,
            model_path=model_path,
            predictions_path=predictions_path,
            runtime_path=runtime_path,
        ),
        summary,
    )


def _execute_runtime(
    *,
    client_datasets: list[ClientData],
    config: FederatedTrainingConfig,
    runtime_plan: FlowerRuntimePlan,
    recorder: FederatedRunRecorder,
    strategy_factory: StrategyFactory,
    initial_parameters: list[np.ndarray],
) -> tuple[list[np.ndarray], str, list[str]]:
    runtime_warnings: list[str] = []

    if runtime_plan.resolved_backend == "debug-sequential":
        recorder.backend = "debug-sequential"
        return (
            _run_debug_sequential_runtime(
                client_datasets=client_datasets,
                config=config,
                recorder=recorder,
                strategy_factory=strategy_factory,
                initial_parameters=initial_parameters,
            ),
            "debug-sequential",
            runtime_warnings,
        )

    try:
        recorder.backend = "ray"
        final_parameters = _run_flower_simulation(
            client_datasets=client_datasets,
            config=config,
            recorder=recorder,
            strategy_factory=strategy_factory,
            initial_parameters=initial_parameters,
        )
        return final_parameters, "ray", runtime_warnings
    except Exception as exc:
        if not config.debug_fallback_on_error:
            raise RuntimeError(
                "Flower Ray simulation failed. The debug sequential runtime was not used "
                "because `debug_fallback_on_error` is disabled. Re-run with "
                "`--debug-fallback-on-error` only if you explicitly want the development "
                "fallback, or inspect the Flower/Ray runtime failure."
            ) from exc
        runtime_warnings.append(
            "Flower Ray simulation failed and the run continued with the explicit "
            "debug sequential fallback because `debug_fallback_on_error` was enabled."
        )
        recorder.backend = "debug-sequential"
        final_parameters = _run_debug_sequential_runtime(
            client_datasets=client_datasets,
            config=config,
            recorder=recorder,
            strategy_factory=strategy_factory,
            initial_parameters=initial_parameters,
        )
        return final_parameters, "debug-sequential", runtime_warnings


def _run_flower_simulation(
    *,
    client_datasets: list[ClientData],
    config: FederatedTrainingConfig,
    recorder: FederatedRunRecorder,
    strategy_factory: StrategyFactory,
    initial_parameters: list[np.ndarray],
) -> list[np.ndarray]:
    require_flower_support()
    data_by_id = {dataset.client_id: dataset for dataset in client_datasets}

    def client_fn(context: fl.common.Context):
        partition_id = int(context.node_config.get("partition-id", context.node_id))
        client = FederatedLogisticRegressionClient(
            data=data_by_id[partition_id],
            model_config=config.model,
            seed=config.seed,
        )
        return client.to_client()

    def server_fn(context: fl.common.Context):
        strategy = strategy_factory.create(initial_parameters, recorder)
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=config.rounds),
        )

    client_app = ClientApp(client_fn)
    server_app = ServerApp(server_fn=server_fn)
    backend_config = {
        "client_resources": dict(config.simulation_resources),
        "init_args": {"ignore_reinit_error": True},
    }
    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=config.num_clients,
        backend_name="ray",
        backend_config=backend_config,
        verbose_logging=False,
    )
    if recorder.final_parameters is None:
        raise RuntimeError("Flower simulation completed without final parameters.")
    return recorder.final_parameters


def _run_debug_sequential_runtime(
    *,
    client_datasets: list[ClientData],
    config: FederatedTrainingConfig,
    recorder: FederatedRunRecorder,
    strategy_factory: StrategyFactory,
    initial_parameters: list[np.ndarray],
) -> list[np.ndarray]:
    require_flower_support()
    strategy = strategy_factory.create(initial_parameters, recorder)
    clients = [
        FederatedLogisticRegressionClient(
            data=dataset,
            model_config=config.model,
            seed=config.seed,
        )
        for dataset in client_datasets
    ]
    parameters = initial_parameters
    rng = np.random.default_rng(config.seed)

    for server_round in range(1, config.rounds + 1):
        fit_sample_size = _sample_size(
            total_clients=len(clients),
            fraction=config.fit_fraction,
            minimum=min(config.min_available_clients, len(clients)),
        )
        fit_indices = rng.choice(len(clients), size=fit_sample_size, replace=False)
        fit_results = []
        for client_index in fit_indices:
            updated_parameters, num_examples, metrics = clients[client_index].fit(parameters, {})
            fit_results.append(
                (
                    None,
                    fl.common.FitRes(
                        status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
                        parameters=fl.common.ndarrays_to_parameters(updated_parameters),
                        num_examples=num_examples,
                        metrics=metrics,
                    ),
                )
            )
        aggregated_parameters, _ = strategy.aggregate_fit(server_round, fit_results, [])
        if aggregated_parameters is None:
            raise RuntimeError("Debug sequential runtime could not aggregate client updates.")
        parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)

        evaluate_sample_size = _sample_size(
            total_clients=len(clients),
            fraction=config.evaluate_fraction,
            minimum=min(config.min_available_clients, len(clients)),
        )
        evaluate_indices = rng.choice(len(clients), size=evaluate_sample_size, replace=False)
        evaluate_results = []
        for client_index in evaluate_indices:
            loss, num_examples, metrics = clients[client_index].evaluate(parameters, {})
            evaluate_results.append(
                (
                    None,
                    fl.common.EvaluateRes(
                        status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
                        loss=loss,
                        num_examples=num_examples,
                        metrics=metrics,
                    ),
                )
            )
        strategy.aggregate_evaluate(server_round, evaluate_results, [])

    recorder.final_parameters = parameters
    return parameters


def _evaluate_and_persist_results(
    *,
    client_datasets: list[ClientData],
    config: FederatedTrainingConfig,
    result_dir: Path,
    recorder: FederatedRunRecorder,
    final_parameters: list[np.ndarray],
    run_id: str,
    actual_backend: str,
) -> tuple[dict[str, Any], Path]:
    per_client: list[dict[str, Any]] = []
    aggregated_inputs: list[tuple[int, dict[str, float]]] = []
    pooled_probabilities: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    pooled_row_ids: list[np.ndarray] = []
    pooled_client_ids: list[np.ndarray] = []

    for dataset in client_datasets:
        model = LogisticRegressionModel(
            n_features=dataset.X_test.shape[1],
            learning_rate=config.model.learning_rate,
            batch_size=config.model.batch_size,
            local_epochs=config.model.epochs,
            l2_regularization=config.model.l2_regularization,
        )
        model.set_parameters(final_parameters)
        probabilities = model.predict_proba(dataset.X_test)
        loss = model.loss(dataset.X_test, dataset.y_test)
        metrics = compute_pooled_classification_metrics(
            y_true=dataset.y_test,
            y_prob=probabilities,
            loss=loss,
        )
        per_client.append(
            {
                "client_id": dataset.client_id,
                "split_name": "client_test",
                "num_examples": int(dataset.y_test.shape[0]),
                "class_balance": summarize_class_balance(dataset.y_test),
                "probability_summary": summarize_probability_distribution(probabilities),
                "metrics": metrics,
            }
        )
        aggregated_inputs.append((int(dataset.y_test.shape[0]), metrics))
        pooled_probabilities.append(probabilities)
        pooled_labels.append(dataset.y_test)
        pooled_row_ids.append(dataset.row_ids_test)
        pooled_client_ids.append(
            np.full(dataset.y_test.shape[0], dataset.client_id, dtype=np.int64)
        )

    aggregated_metrics = aggregate_weighted_metrics(aggregated_inputs)
    pooled_y_true = np.concatenate(pooled_labels, axis=0)
    pooled_y_prob = np.concatenate(pooled_probabilities, axis=0)
    pooled_loss = float(
        np.average(
            [row["metrics"]["loss"] for row in per_client],
            weights=[row["num_examples"] for row in per_client],
        )
    )
    pooled_metrics = compute_pooled_classification_metrics(
        y_true=pooled_y_true,
        y_prob=pooled_y_prob,
        loss=pooled_loss,
    )
    threshold_sweep = sweep_classification_thresholds(y_true=pooled_y_true, y_prob=pooled_y_prob)
    predictions = build_prediction_artifact(
        run_id=run_id,
        dataset_name=config.dataset_name,
        split_name="client_test",
        y_true=pooled_y_true,
        y_prob=pooled_y_prob,
        row_ids=np.concatenate(pooled_row_ids, axis=0),
        client_ids=np.concatenate(pooled_client_ids, axis=0),
    )
    predictions_path = save_prediction_artifact(result_dir / "predictions_client_test.npz", predictions)

    summary = {
        "run_id": run_id,
        "experiment_type": "federated",
        "dataset_name": config.dataset_name,
        "result_dir": str(result_dir),
        "config": config.to_dict(),
        "round_history": recorder.round_history,
        "evaluation": {
            "client_test_weighted": {
                "split_name": "client_test_weighted",
                "provenance": {
                    "source": "weighted_client_evaluation",
                    "client_count": len(client_datasets),
                    "backend": actual_backend,
                },
                "class_balance": summarize_class_balance(pooled_y_true),
                "probability_summary": summarize_probability_distribution(pooled_y_prob),
                "metrics": aggregated_metrics,
            },
            "client_test_pooled": {
                "split_name": "client_test",
                "provenance": {
                    "source": "pooled_client_predictions",
                    "client_count": len(client_datasets),
                    "backend": actual_backend,
                },
                "class_balance": summarize_class_balance(pooled_y_true),
                "probability_summary": summarize_probability_distribution(pooled_y_prob),
                "metrics": pooled_metrics,
                "predictions_path": str(predictions_path),
            },
            "per_client": per_client,
        },
        "per_client": per_client,
        "aggregated_weighted": aggregated_metrics,
        "aggregated_pooled": pooled_metrics,
        "threshold_analysis": threshold_sweep,
        "final_parameters": [final_parameters[0].tolist(), final_parameters[1].tolist()],
        "predictions_path": str(predictions_path),
    }
    return summary, predictions_path


def _canonicalize_backend(requested_backend: str) -> str:
    try:
        return BACKEND_ALIASES[requested_backend]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKEND_ALIASES))
        raise ValueError(
            f"Unsupported simulation backend '{requested_backend}'. Supported backends: {supported}."
        ) from exc


def _sample_size(total_clients: int, fraction: float, minimum: int) -> int:
    sample_size = int(np.ceil(total_clients * fraction))
    return max(1, min(total_clients, max(minimum, sample_size)))


def flower_support_available() -> bool:
    """Return whether Flower is importable in the current environment."""

    return fl is not None


def require_flower_support() -> None:
    """Raise a clear error when Flower support is unavailable."""

    if fl is None:  # pragma: no cover - depends on optional deps
        raise ImportError(FLOWER_IMPORT_ERROR_MESSAGE)
