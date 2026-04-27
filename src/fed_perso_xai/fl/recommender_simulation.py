"""Flower runtime integration for federated recommender training."""

from __future__ import annotations

from typing import Any

import logging
import numpy as np

from fed_perso_xai.fl.client import FederatedPairwiseRecommenderClient, RecommenderClientData
from fed_perso_xai.fl.simulation import (
    ClientApp,
    ServerApp,
    ServerAppComponents,
    ServerConfig,
    SimulationArtifacts,
    _sample_size,
    fl,
    plan_flower_runtime,
    require_flower_support,
)
from fed_perso_xai.fl.strategy import FederatedRunRecorder, StrategyFactory, create_strategy_factory
from fed_perso_xai.recommender import PairwiseLogisticConfig, initialize_recommender_parameters
from fed_perso_xai.utils.config import RecommenderFederatedTrainingConfig

LOGGER = logging.getLogger(__name__)


def run_federated_recommender_training(
    *,
    client_datasets: list[RecommenderClientData],
    config: RecommenderFederatedTrainingConfig,
    model_config: PairwiseLogisticConfig,
    strategy_factory: StrategyFactory | None = None,
) -> SimulationArtifacts:
    """Run federated pairwise recommender training."""

    require_flower_support()
    if not client_datasets:
        raise ValueError("client_datasets must not be empty.")

    runtime_config = config.with_num_clients(len(client_datasets))
    runtime_plan = plan_flower_runtime(runtime_config)  # type: ignore[arg-type]
    LOGGER.info(
        "Starting federated recommender training clients=%s rounds=%s backend_requested=%s backend_planned=%s total_train_pairs=%s total_eval_pairs=%s",
        len(client_datasets),
        runtime_config.rounds,
        runtime_plan.requested_backend,
        runtime_plan.resolved_backend,
        int(sum(dataset.y_train.shape[0] for dataset in client_datasets)),
        int(sum(dataset.y_eval.shape[0] for dataset in client_datasets)),
    )
    initial_parameters = initialize_recommender_parameters(client_datasets[0].X_train.shape[1])
    recorder = FederatedRunRecorder(backend=runtime_plan.resolved_backend)
    factory = strategy_factory or create_strategy_factory(
        runtime_config.strategy_name,
        training_config=runtime_config,  # type: ignore[arg-type]
    )
    final_parameters, actual_backend, runtime_warnings = _execute_recommender_runtime(
        client_datasets=client_datasets,
        config=runtime_config,
        model_config=model_config,
        recorder=recorder,
        strategy_factory=factory,
        initial_parameters=initial_parameters,
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
    LOGGER.info(
        "Completed federated recommender training rounds_completed=%s actual_backend=%s",
        len(recorder.round_history),
        actual_backend,
    )
    return SimulationArtifacts(
        final_parameters=[np.asarray(parameter, dtype=np.float64).copy() for parameter in final_parameters],
        round_history=[dict(item) for item in recorder.round_history],
        runtime_report=runtime_report,
        actual_backend=actual_backend,
    )


def _execute_recommender_runtime(
    *,
    client_datasets: list[RecommenderClientData],
    config: RecommenderFederatedTrainingConfig,
    model_config: PairwiseLogisticConfig,
    recorder: FederatedRunRecorder,
    strategy_factory: StrategyFactory,
    initial_parameters: list[np.ndarray],
) -> tuple[list[np.ndarray], str, list[str]]:
    runtime_warnings: list[str] = []
    runtime_plan = plan_flower_runtime(config)  # type: ignore[arg-type]

    if runtime_plan.resolved_backend == "debug-sequential":
        recorder.backend = "debug-sequential"
        return (
            _run_debug_sequential_recommender_runtime(
                client_datasets=client_datasets,
                config=config,
                model_config=model_config,
                recorder=recorder,
                strategy_factory=strategy_factory,
                initial_parameters=initial_parameters,
            ),
            "debug-sequential",
            runtime_warnings,
        )

    try:
        recorder.backend = "ray"
        final_parameters = _run_flower_recommender_simulation(
            client_datasets=client_datasets,
            config=config,
            model_config=model_config,
            recorder=recorder,
            strategy_factory=strategy_factory,
            initial_parameters=initial_parameters,
        )
        return final_parameters, "ray", runtime_warnings
    except Exception as exc:
        if not config.debug_fallback_on_error:
            raise RuntimeError(
                "Flower Ray recommender simulation failed. Re-run with "
                "--debug-fallback-on-error only if you explicitly want the development fallback."
            ) from exc
        runtime_warnings.append(
            "Flower Ray recommender simulation failed and continued with the explicit debug sequential fallback."
        )
        recorder.backend = "debug-sequential"
        final_parameters = _run_debug_sequential_recommender_runtime(
            client_datasets=client_datasets,
            config=config,
            model_config=model_config,
            recorder=recorder,
            strategy_factory=strategy_factory,
            initial_parameters=initial_parameters,
        )
        return final_parameters, "debug-sequential", runtime_warnings


def _run_flower_recommender_simulation(
    *,
    client_datasets: list[RecommenderClientData],
    config: RecommenderFederatedTrainingConfig,
    model_config: PairwiseLogisticConfig,
    recorder: FederatedRunRecorder,
    strategy_factory: StrategyFactory,
    initial_parameters: list[np.ndarray],
) -> list[np.ndarray]:
    require_flower_support()
    data_by_id = {dataset.client_id: dataset for dataset in client_datasets}

    def client_fn(context: Any):
        partition_id = int(context.node_config.get("partition-id", context.node_id))
        client = FederatedPairwiseRecommenderClient(
            data=data_by_id[partition_id],
            model_config=model_config,
            seed=config.seed,
        )
        return client.to_client()

    def server_fn(context: Any):
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
        num_supernodes=len(client_datasets),
        backend_name="ray",
        backend_config=backend_config,
        verbose_logging=False,
    )
    if recorder.final_parameters is None:
        raise RuntimeError("Flower recommender simulation completed without final parameters.")
    return recorder.final_parameters


def _run_debug_sequential_recommender_runtime(
    *,
    client_datasets: list[RecommenderClientData],
    config: RecommenderFederatedTrainingConfig,
    model_config: PairwiseLogisticConfig,
    recorder: FederatedRunRecorder,
    strategy_factory: StrategyFactory,
    initial_parameters: list[np.ndarray],
) -> list[np.ndarray]:
    require_flower_support()
    strategy = strategy_factory.create(initial_parameters, recorder)
    clients = [
        FederatedPairwiseRecommenderClient(
            data=dataset,
            model_config=model_config,
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
            raise RuntimeError("Debug sequential recommender runtime could not aggregate client updates.")
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
