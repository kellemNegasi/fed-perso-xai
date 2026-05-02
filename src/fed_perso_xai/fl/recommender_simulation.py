"""Flower runtime integration for federated recommender training."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from fed_perso_xai.fl.client import FederatedPairwiseRecommenderClient, RecommenderClientData
from fed_perso_xai.fl.simulation import (
    ClientApp,
    ServerApp,
    ServerAppComponents,
    ServerConfig,
    _sample_size,
    fl,
    plan_flower_runtime,
    require_flower_support,
)
from fed_perso_xai.fl.strategy import FederatedRunRecorder, StrategyFactory, create_strategy_factory
from fed_perso_xai.recommender import (
    PairwiseLogisticConfig,
    create_recommender,
    initialize_recommender_parameters,
)
from fed_perso_xai.recommender.clustering import (
    ClientSideRandomProjector,
    IdentityProjectionSpec,
    PCAProjectionSpec,
    RecommenderWeightVectorExtractor,
    SecureClusterModelAggregator,
    SecureKMeansClusterer,
    build_centered_pca_projection_spec,
    build_identity_projection_spec,
    summarize_cluster_sizes,
    weighted_average_parameter_sets,
)
from fed_perso_xai.utils.config import RecommenderFederatedTrainingConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusteredRoundResult:
    """Round-scoped clustered recommender outputs retained for artifact writing."""

    round_id: int
    assignments: dict[str, int]
    cluster_sizes: dict[str, int]
    projection_metadata: dict[str, Any]
    secure_clustering_metadata: dict[str, Any]
    secure_aggregation_metadata: dict[str, dict[str, Any]]
    cluster_parameters: dict[int, list[np.ndarray]]


@dataclass(frozen=True)
class RecommenderSimulationArtifacts:
    """In-memory training outputs for one recommender federated run."""

    final_parameters: list[np.ndarray]
    round_history: list[dict[str, object]]
    runtime_report: dict[str, Any]
    actual_backend: str
    clustered: bool = False
    clustered_rounds: list[ClusteredRoundResult] = field(default_factory=list)
    final_cluster_assignments: dict[str, int] = field(default_factory=dict)
    final_cluster_parameters: dict[int, list[np.ndarray]] = field(default_factory=dict)


def _align_cluster_labels_to_previous_round(
    *,
    previous_assignments: Mapping[str, int],
    current_assignments: Mapping[str, int],
    cluster_count: int,
) -> tuple[dict[int, int], int]:
    """Align current cluster ids to the previous round by maximum client overlap."""

    identity_mapping = {cluster_id: cluster_id for cluster_id in range(cluster_count)}
    shared_clients = sorted(set(previous_assignments).intersection(current_assignments))
    if not shared_clients:
        return identity_mapping, 0

    overlap_matrix = np.zeros((cluster_count, cluster_count), dtype=np.int64)
    for client_id in shared_clients:
        previous_cluster_id = int(previous_assignments[client_id])
        current_cluster_id = int(current_assignments[client_id])
        overlap_matrix[previous_cluster_id, current_cluster_id] += 1

    # Break ties deterministically in favor of lexicographically smaller
    # previous-label assignments while keeping the primary objective as
    # maximum overlap.
    tie_break_bias = np.fromfunction(
        lambda previous_cluster_id, current_cluster_id: -(
            current_cluster_id.astype(np.int64) * cluster_count + previous_cluster_id.astype(np.int64)
        ),
        overlap_matrix.shape,
        dtype=np.int64,
    )
    row_indices, column_indices = linear_sum_assignment(
        overlap_matrix * (cluster_count**2) + tie_break_bias,
        maximize=True,
    )
    aligned_pairs = sorted(
        (
            int(current_cluster_id),
            int(previous_cluster_id),
        )
        for previous_cluster_id, current_cluster_id in zip(row_indices, column_indices, strict=True)
    )
    mapping = {current_cluster_id: previous_cluster_id for current_cluster_id, previous_cluster_id in aligned_pairs}
    overlap = sum(int(overlap_matrix[previous_cluster_id, current_cluster_id]) for current_cluster_id, previous_cluster_id in aligned_pairs)
    return mapping, overlap


def _count_assignment_changes(
    *,
    previous_assignments: Mapping[str, int],
    current_assignments: Mapping[str, int],
) -> tuple[int, int]:
    """Count assignment changes across the shared clients of two rounds."""

    shared_clients = sorted(set(previous_assignments).intersection(current_assignments))
    if not shared_clients:
        return 0, 0
    changed = sum(
        int(previous_assignments[client_id]) != int(current_assignments[client_id]) for client_id in shared_clients
    )
    return int(changed), int(len(shared_clients))


def run_federated_recommender_training(
    *,
    client_datasets: list[RecommenderClientData],
    config: RecommenderFederatedTrainingConfig,
    model_config: PairwiseLogisticConfig,
    strategy_factory: StrategyFactory | None = None,
) -> RecommenderSimulationArtifacts:
    """Run federated pairwise recommender training."""

    if not client_datasets:
        raise ValueError("client_datasets must not be empty.")

    runtime_config = config.with_num_clients(len(client_datasets))
    if runtime_config.clustering.enabled:
        return _run_clustered_recommender_training(
            client_datasets=client_datasets,
            config=runtime_config,
            model_config=model_config,
        )

    require_flower_support()
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
    initial_parameters = initialize_recommender_parameters(
        client_datasets[0].X_train.shape[1],
        recommender_type=config.recommender_type,
    )
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
    return RecommenderSimulationArtifacts(
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
            recommender_type=config.recommender_type,
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
        "init_args": {
            "ignore_reinit_error": True,
            "num_cpus": config.ray_num_cpus,
        },
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
            recommender_type=config.recommender_type,
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

        if float(config.evaluate_fraction) <= 0.0:
            continue

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


def _run_clustered_recommender_training(
    *,
    client_datasets: list[RecommenderClientData],
    config: RecommenderFederatedTrainingConfig,
    model_config: PairwiseLogisticConfig,
) -> RecommenderSimulationArtifacts:
    clustering_config = config.clustering
    if not clustering_config.enabled:
        raise ValueError("Clustered recommender runtime requires clustering.enabled=True.")
    if clustering_config.k > len(client_datasets):
        raise ValueError("clustering.k cannot exceed the number of recommender clients.")

    initial_parameters = initialize_recommender_parameters(
        client_datasets[0].X_train.shape[1],
        recommender_type=config.recommender_type,
    )
    extractor = RecommenderWeightVectorExtractor()
    projector = ClientSideRandomProjector(config)
    clusterer = SecureKMeansClusterer(config)
    secure_aggregator = SecureClusterModelAggregator(config)
    previous_assignments: dict[str, int] = {}
    warmup_rounds = int(clustering_config.warmup_rounds)
    freeze_pca_after_warmup = bool(clustering_config.freeze_pca_after_warmup)
    use_pca = bool(clustering_config.enable_pca)
    frozen_projection_spec: PCAProjectionSpec | IdentityProjectionSpec | None = None
    shared_global_parameters = [np.asarray(parameter, dtype=np.float64).copy() for parameter in initial_parameters]
    current_cluster_models = {
        cluster_id: [np.asarray(parameter, dtype=np.float64).copy() for parameter in shared_global_parameters]
        for cluster_id in range(clustering_config.k)
    }
    round_history: list[dict[str, object]] = []
    clustered_rounds: list[ClusteredRoundResult] = []

    warnings = [
        "Clustered recommender training uses the in-process clustered sequential runtime.",
        "Clustered recommender training currently uses full client participation in every round.",
    ]
    if config.secure_aggregation:
        warnings.append(
            "The recommender clustering path always performs secure per-cluster aggregation; the top-level secure_aggregation flag remains relevant for the non-clustered FedAvg path only."
        )

    LOGGER.info(
        "Starting clustered recommender training clients=%s rounds=%s clusters=%s warmup_rounds=%s freeze_pca_after_warmup=%s recommender_type=%s",
        len(client_datasets),
        config.rounds,
        clustering_config.k,
        warmup_rounds,
        freeze_pca_after_warmup,
        config.recommender_type,
    )

    for server_round in range(1, config.rounds + 1):
        local_parameters: dict[str, list[np.ndarray]] = {}
        local_weights: dict[str, int] = {}
        train_losses: dict[str, float] = {}
        is_warmup_round = server_round <= warmup_rounds

        for dataset in client_datasets:
            client_name = dataset.client_name
            if is_warmup_round:
                starting_parameters = shared_global_parameters
            else:
                starting_parameters = current_cluster_models.get(
                    previous_assignments.get(client_name, 0),
                    shared_global_parameters,
                )
            fitted_parameters, train_loss = _fit_local_recommender(
                dataset=dataset,
                model_config=model_config,
                recommender_type=config.recommender_type,
                seed=config.seed,
                base_parameters=starting_parameters,
            )
            local_parameters[client_name] = fitted_parameters
            local_weights[client_name] = int(dataset.y_train.shape[0])
            train_losses[client_name] = float(train_loss)

        weighted_train_loss = _weighted_scalar_average(train_losses, local_weights)
        if is_warmup_round:
            ordered_client_names = sorted(local_parameters)
            shared_global_parameters = weighted_average_parameter_sets(
                [local_parameters[client_name] for client_name in ordered_client_names],
                [local_weights[client_name] for client_name in ordered_client_names],
            )
            current_cluster_models = {
                cluster_id: [np.asarray(parameter, dtype=np.float64).copy() for parameter in shared_global_parameters]
                for cluster_id in range(clustering_config.k)
            }
            LOGGER.info(
                "Clustered recommender warmup round=%s/%s train_loss_weighted=%.6f",
                int(server_round),
                warmup_rounds,
                float(weighted_train_loss),
            )
            round_history.append(
                {
                    "round": int(server_round),
                    "fit_metrics": {
                        "train_loss_weighted": float(weighted_train_loss),
                    },
                    "aggregation": {
                        "mode": "warmup_global",
                        "num_clusters": int(clustering_config.k),
                        "num_contributors": int(len(local_parameters)),
                    },
                    "evaluate_skipped": True,
                }
            )
            continue

        ordered_local_parameters = {
            client_name: local_parameters[client_name]
            for client_name in sorted(local_parameters)
        }
        projection_fitted_this_round = False
        if not use_pca:
            projection_dimension = int(extractor.flatten(next(iter(ordered_local_parameters.values()))).shape[0])
            projection_spec = build_identity_projection_spec(input_dimension=projection_dimension)
            projection_generation_mode = "identity_no_projection"
            projection_server_observes_raw_weights = False
        elif freeze_pca_after_warmup and frozen_projection_spec is not None:
            projection_spec = frozen_projection_spec
            projection_generation_mode = "frozen_after_warmup_reuse"
            projection_server_observes_raw_weights = False
        else:
            _, flattened_vectors = extractor.flatten_many(ordered_local_parameters)
            projection_spec = build_centered_pca_projection_spec(
                flattened_vectors=flattened_vectors,
                requested_components=clustering_config.pca_components,
                seed=int(config.seed + server_round - 1),
            )
            projection_fitted_this_round = True
            if freeze_pca_after_warmup:
                frozen_projection_spec = projection_spec
                projection_generation_mode = "server_fit_once_after_warmup_then_frozen"
            else:
                projection_generation_mode = "server_fit_from_centered_local_models_per_round"
            projection_server_observes_raw_weights = True
        shared_reduced_vectors = [
            projector.build_private_reduced_vector(
                client_id=client_name,
                parameters=ordered_local_parameters[client_name],
                projection_spec=projection_spec,
                round_id=server_round,
            )
            for client_name in ordered_local_parameters
        ]
        clustering_seed = int(config.seed + server_round - 1)
        assignments_result = clusterer.cluster(
            shared_reduced_vectors,
            projection_spec=projection_spec,
            seed=clustering_seed,
            clustering_config=clustering_config,
        )
        raw_assignments = {
            shared_vector.client_id: int(label)
            for shared_vector, label in zip(shared_reduced_vectors, assignments_result.labels, strict=True)
        }
        # Raw clustering labels are arbitrary each round. Remap them to the
        # previous round's cluster ids so the carried-forward cluster models
        # keep a stable identity across rounds.
        label_alignment, label_alignment_overlap = _align_cluster_labels_to_previous_round(
            previous_assignments=previous_assignments,
            current_assignments=raw_assignments,
            cluster_count=clustering_config.k,
        )
        assignments = {
            client_id: int(label_alignment[cluster_id]) for client_id, cluster_id in raw_assignments.items()
        }
        cluster_sizes = summarize_cluster_sizes(assignments, clustering_config.k)
        changed_clients, compared_clients = _count_assignment_changes(
            previous_assignments=previous_assignments,
            current_assignments=assignments,
        )
        aggregated_clusters = secure_aggregator.aggregate(
            client_parameters=local_parameters,
            client_weights=local_weights,
            assignments=assignments,
            round_id=server_round,
            cluster_count=clustering_config.k,
            fallback_parameters=current_cluster_models,
        )
        current_cluster_models = {
            cluster_id: [
                np.asarray(parameter, dtype=np.float64).copy()
                for parameter in aggregated_clusters[cluster_id].parameters
            ]
            for cluster_id in range(clustering_config.k)
        }
        previous_assignments = dict(assignments)

        LOGGER.info(
            "Clustered recommender round=%s sizes=%s changed_clients=%s/%s label_alignment_overlap=%s train_loss_weighted=%.6f",
            int(server_round),
            dict(cluster_sizes),
            int(changed_clients),
            int(compared_clients),
            int(label_alignment_overlap),
            float(weighted_train_loss),
        )

        round_history.append(
            {
                "round": int(server_round),
                "fit_metrics": {
                    "train_loss_weighted": float(weighted_train_loss),
                },
                "aggregation": {
                    "mode": "clustered_secure",
                    "num_clusters": int(clustering_config.k),
                    "cluster_sizes": dict(cluster_sizes),
                    "num_contributors": int(len(local_parameters)),
                },
                "evaluate_skipped": True,
            }
        )
        clustered_rounds.append(
            ClusteredRoundResult(
                round_id=int(server_round),
                assignments=dict(assignments),
                cluster_sizes=dict(cluster_sizes),
                projection_metadata={
                    **projection_spec.to_metadata(),
                    "projection_generation_mode": projection_generation_mode,
                    "server_observes_raw_weights_during_projection_fit": projection_server_observes_raw_weights,
                    "projection_fit_round_id": int(server_round) if projection_fitted_this_round else None,
                    "warmup_rounds": warmup_rounds,
                    "freeze_pca_after_warmup": freeze_pca_after_warmup,
                    "enable_pca": use_pca,
                },
                secure_clustering_metadata={
                    **assignments_result.secure_metadata,
                    "label_alignment_to_previous_round": {
                        str(cluster_id): int(aligned_cluster_id)
                        for cluster_id, aligned_cluster_id in label_alignment.items()
                    },
                    "label_alignment_overlap_count": int(label_alignment_overlap),
                    "initial_centroid_indices": [
                        int(value) for value in assignments_result.initial_centroid_indices
                    ],
                    "centroid_shape": [int(value) for value in assignments_result.centroids.shape],
                },
                secure_aggregation_metadata={
                    str(cluster_id): dict(result.metadata)
                    for cluster_id, result in aggregated_clusters.items()
                },
                cluster_parameters={
                    cluster_id: [
                        np.asarray(parameter, dtype=np.float64).copy()
                        for parameter in result.parameters
                    ]
                    for cluster_id, result in aggregated_clusters.items()
                },
            )
        )

    final_cluster_assignments = dict(previous_assignments)
    final_cluster_parameters = {
        cluster_id: [np.asarray(parameter, dtype=np.float64).copy() for parameter in parameters]
        for cluster_id, parameters in current_cluster_models.items()
    }
    non_empty_cluster_ids = [
        cluster_id
        for cluster_id in range(clustering_config.k)
        if int(clustered_rounds[-1].cluster_sizes[str(cluster_id)]) > 0
    ]
    if not non_empty_cluster_ids:
        raise RuntimeError("Clustered recommender training completed without any populated clusters.")
    final_parameters = weighted_average_parameter_sets(
        [final_cluster_parameters[cluster_id] for cluster_id in non_empty_cluster_ids],
        [int(clustered_rounds[-1].cluster_sizes[str(cluster_id)]) for cluster_id in non_empty_cluster_ids],
    )

    actual_backend = "clustered-sequential"
    server_observes_raw_weights_during_clustering = any(
        bool(round_result.projection_metadata.get("server_observes_raw_weights_during_projection_fit", False))
        for round_result in clustered_rounds
    )
    runtime_report = {
        "requested_backend": config.simulation_backend,
        "planned_backend": actual_backend,
        "actual_backend": actual_backend,
        "ray_available": importlib.util.find_spec("ray") is not None,
        "flwr_version": _optional_package_version("flwr"),
        "is_debug_runtime": True,
        "warnings": warnings,
        "rounds_completed": len(round_history),
        "clustered": True,
        "cluster_count": int(clustering_config.k),
        "warmup_rounds": warmup_rounds,
        "freeze_pca_after_warmup": freeze_pca_after_warmup,
        "server_observes_raw_weights_during_clustering": server_observes_raw_weights_during_clustering,
    }
    LOGGER.info(
        "Completed clustered recommender training rounds_completed=%s actual_backend=%s",
        len(round_history),
        actual_backend,
    )
    return RecommenderSimulationArtifacts(
        final_parameters=[np.asarray(parameter, dtype=np.float64).copy() for parameter in final_parameters],
        round_history=[dict(item) for item in round_history],
        runtime_report=runtime_report,
        actual_backend=actual_backend,
        clustered=True,
        clustered_rounds=clustered_rounds,
        final_cluster_assignments=final_cluster_assignments,
        final_cluster_parameters=final_cluster_parameters,
    )


def _fit_local_recommender(
    *,
    dataset: RecommenderClientData,
    model_config: PairwiseLogisticConfig,
    recommender_type: str,
    seed: int,
    base_parameters: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], float]:
    model = create_recommender(
        recommender_type=recommender_type,
        n_features=dataset.X_train.shape[1],
        config=model_config,
    )
    model.set_parameters(base_parameters)
    train_loss = model.fit(
        dataset.X_train,
        dataset.y_train,
        seed=int(seed + dataset.client_id),
    )
    return model.get_parameters(), float(train_loss)


def _weighted_scalar_average(
    values: Mapping[str, float],
    weights: Mapping[str, int | float],
) -> float:
    total_weight = float(sum(float(weights[key]) for key in values))
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return float(
        sum(float(values[key]) * float(weights[key]) for key in values) / total_weight
    )


def _optional_package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"
