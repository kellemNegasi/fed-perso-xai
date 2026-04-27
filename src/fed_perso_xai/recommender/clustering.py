"""Helpers for clustered federated recommender training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fed_perso_xai.utils.config import RecommenderClusteringConfig, RecommenderFederatedTrainingConfig


@dataclass(frozen=True)
class PCAProjectionResult:
    """PCA-reduced client weight vectors plus reproducibility metadata."""

    reduced_vectors: np.ndarray
    mean_vector: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    requested_components: int
    actual_components: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "requested_components": int(self.requested_components),
            "actual_components": int(self.actual_components),
            "input_dimension": int(self.mean_vector.shape[0]),
            "component_matrix_shape": [int(value) for value in self.components.shape],
            "explained_variance": self.explained_variance.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "mean_vector": self.mean_vector.tolist(),
        }


@dataclass(frozen=True)
class SecureClusterAssignments:
    """Assignments and metadata from one secure clustering round."""

    labels: np.ndarray
    centroids: np.ndarray
    iterations: int
    initial_centroid_indices: tuple[int, ...]
    secure_metadata: dict[str, Any]


@dataclass(frozen=True)
class ClusterAggregationResult:
    """One cluster model after secure aggregation."""

    parameters: list[np.ndarray]
    metadata: dict[str, Any]


class RecommenderWeightVectorExtractor:
    """Flatten full recommender parameter payloads into one vector per client."""

    def flatten(self, parameters: Sequence[np.ndarray]) -> np.ndarray:
        arrays = [np.asarray(parameter, dtype=np.float64).reshape(-1) for parameter in parameters]
        if not arrays:
            raise ValueError("parameters must contain at least one tensor.")
        return np.concatenate(arrays, axis=0)

    def flatten_many(
        self,
        parameter_sets: Mapping[str, Sequence[np.ndarray]],
    ) -> tuple[list[str], np.ndarray]:
        client_ids = list(parameter_sets)
        if not client_ids:
            raise ValueError("parameter_sets must not be empty.")
        flattened = [self.flatten(parameter_sets[client_id]) for client_id in client_ids]
        reference_dimension = flattened[0].shape[0]
        for vector in flattened[1:]:
            if vector.shape[0] != reference_dimension:
                raise ValueError("All recommender payloads must share the same flattened size.")
        return client_ids, np.stack(flattened, axis=0)


class PCAReducer:
    """Project flattened recommender weights into a lower-dimensional PCA space."""

    def reduce(self, weight_vectors: np.ndarray, requested_components: int) -> PCAProjectionResult:
        vectors = np.asarray(weight_vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError("weight_vectors must be a 2D array.")
        if vectors.shape[0] == 0:
            raise ValueError("weight_vectors must contain at least one row.")

        actual_components = max(1, min(int(requested_components), vectors.shape[0], vectors.shape[1]))
        mean_vector = np.mean(vectors, axis=0)
        centered = vectors - mean_vector
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:actual_components].copy()
        reduced = centered @ components.T
        if vectors.shape[0] > 1:
            explained_variance = (singular_values[:actual_components] ** 2) / float(vectors.shape[0] - 1)
            total_variance = (singular_values**2).sum() / float(vectors.shape[0] - 1)
        else:
            explained_variance = np.zeros(actual_components, dtype=np.float64)
            total_variance = 0.0
        if total_variance > 0.0:
            explained_variance_ratio = explained_variance / total_variance
        else:
            explained_variance_ratio = np.zeros(actual_components, dtype=np.float64)
        return PCAProjectionResult(
            reduced_vectors=reduced,
            mean_vector=mean_vector,
            components=components,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            requested_components=int(requested_components),
            actual_components=int(actual_components),
        )


class SecureKMeansClusterer:
    """Secure K-means clustering over PCA-reduced recommender weights."""

    def __init__(self, training_config: RecommenderFederatedTrainingConfig) -> None:
        self.training_config = training_config

    def cluster(
        self,
        reduced_vectors: np.ndarray,
        *,
        seed: int,
        clustering_config: RecommenderClusteringConfig,
    ) -> SecureClusterAssignments:
        features = np.asarray(reduced_vectors, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError("reduced_vectors must be a 2D array.")
        if features.shape[0] < clustering_config.k:
            raise ValueError("clustering.k cannot exceed the number of participating clients.")

        clustering, protocol_quantization, simulation_config, lcc_config = (
            _build_secure_clustering_components(self.training_config, clustering_config, seed)
        )
        rng = np.random.default_rng(seed)
        initial_indices = tuple(
            int(value)
            for value in rng.choice(features.shape[0], size=clustering_config.k, replace=False)
        )
        initial_centroids = features[np.asarray(initial_indices, dtype=np.int64)].copy()
        result = clustering.fit_secure(features=features, initial_centroids=initial_centroids)
        last_trace = clustering.aggregator.last_trace
        secure_metadata = {
            "method": clustering_config.method,
            "seed": int(seed),
            "iterations": int(result.iterations),
            "n_clusters": int(clustering_config.k),
            "max_iterations": int(clustering_config.max_iterations),
            "tolerance": float(clustering_config.tolerance),
            "helper_count": int(lcc_config.n_helpers),
            "privacy_threshold": int(lcc_config.privacy_threshold),
            "reconstruction_threshold": int(lcc_config.resolved_reconstruction_threshold),
            "field_modulus": int(protocol_quantization.prime),
            "quantization_scale": int(protocol_quantization.scale),
            "helper_ids": list(last_trace.helper_ids) if last_trace is not None else [],
            "helper_evaluation_points": (
                list(last_trace.evaluation_points) if last_trace is not None else []
            ),
        }
        return SecureClusterAssignments(
            labels=result.labels.astype(np.int64, copy=False),
            centroids=result.centroids.astype(np.float64, copy=True),
            iterations=int(result.iterations),
            initial_centroid_indices=initial_indices,
            secure_metadata=secure_metadata,
        )


class SecureClusterModelAggregator:
    """Securely aggregate recommender models within each cluster."""

    def __init__(self, training_config: RecommenderFederatedTrainingConfig) -> None:
        self.training_config = training_config

    def aggregate(
        self,
        *,
        client_parameters: Mapping[str, Sequence[np.ndarray]],
        client_weights: Mapping[str, int | float],
        assignments: Mapping[str, int],
        round_id: int,
        cluster_count: int,
        fallback_parameters: Mapping[int, Sequence[np.ndarray]],
    ) -> dict[int, ClusterAggregationResult]:
        aggregator = _build_secure_round_aggregator(self.training_config)
        results: dict[int, ClusterAggregationResult] = {}
        for cluster_id in range(cluster_count):
            member_ids = [
                client_id
                for client_id, assigned_cluster in assignments.items()
                if int(assigned_cluster) == cluster_id
            ]
            if not member_ids:
                fallback = fallback_parameters.get(cluster_id)
                if fallback is None:
                    raise ValueError(f"Missing fallback parameters for empty cluster {cluster_id}.")
                results[cluster_id] = ClusterAggregationResult(
                    parameters=[np.asarray(parameter, dtype=np.float64).copy() for parameter in fallback],
                    metadata={
                        "cluster_id": int(cluster_id),
                        "round_id": int(round_id),
                        "mode": "carry_forward_empty_cluster",
                        "num_contributors": 0,
                        "client_ids": [],
                    },
                )
                continue

            total_weight = float(sum(float(client_weights[client_id]) for client_id in member_ids))
            if total_weight <= 0.0:
                raise ValueError(f"Cluster {cluster_id} requires a positive total client weight.")
            weighted_payloads = [
                _scale_parameter_set(
                    client_parameters[client_id],
                    float(client_weights[client_id]) / total_weight,
                )
                for client_id in member_ids
            ]
            secure_result = aggregator.aggregate(
                weighted_payloads,
                round_id=(int(round_id) * 10_000) + cluster_id,
                client_ids=list(member_ids),
            )
            results[cluster_id] = ClusterAggregationResult(
                parameters=[
                    np.asarray(parameter, dtype=np.float64).copy()
                    for parameter in secure_result.aggregated_tensors
                ],
                metadata={
                    "cluster_id": int(cluster_id),
                    "round_id": int(round_id),
                    "mode": "secure",
                    "client_ids": list(member_ids),
                    "num_contributors": int(secure_result.num_contributors),
                    "helper_count": int(len(secure_result.helper_ids)),
                    "helper_ids": [int(value) for value in secure_result.helper_ids],
                    "helper_evaluation_points": [
                        int(value) for value in secure_result.helper_evaluation_points
                    ],
                    "field_modulus": int(self.training_config.secure_field_modulus),
                    "quantization_scale": int(self.training_config.secure_quantization_scale),
                    "max_abs_error": float(secure_result.max_abs_error),
                },
            )
        return results


def summarize_cluster_sizes(assignments: Mapping[str, int], cluster_count: int) -> dict[str, int]:
    counts = {str(cluster_id): 0 for cluster_id in range(cluster_count)}
    for cluster_id in assignments.values():
        counts[str(int(cluster_id))] = counts.get(str(int(cluster_id)), 0) + 1
    return counts


def weighted_average_parameter_sets(
    parameter_sets: Sequence[Sequence[np.ndarray]],
    weights: Sequence[int | float],
) -> list[np.ndarray]:
    if not parameter_sets:
        raise ValueError("parameter_sets must contain at least one payload.")
    if len(parameter_sets) != len(weights):
        raise ValueError("weights must align with parameter_sets.")

    reference_shapes = tuple(tuple(np.asarray(array).shape) for array in parameter_sets[0])
    weighted_sums = [
        np.zeros_like(np.asarray(parameter, dtype=np.float64), dtype=np.float64)
        for parameter in parameter_sets[0]
    ]
    total_weight = float(sum(float(weight) for weight in weights))
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive value.")

    for payload, weight in zip(parameter_sets, weights, strict=True):
        payload_shapes = tuple(tuple(np.asarray(array).shape) for array in payload)
        if payload_shapes != reference_shapes:
            raise ValueError("All parameter payloads must share the same shapes.")
        for index, parameter in enumerate(payload):
            weighted_sums[index] += np.asarray(parameter, dtype=np.float64) * float(weight)
    return [parameter / total_weight for parameter in weighted_sums]


def _scale_parameter_set(
    parameters: Sequence[np.ndarray],
    factor: int | float,
) -> list[np.ndarray]:
    return [np.asarray(parameter, dtype=np.float64) * float(factor) for parameter in parameters]


def _build_secure_round_aggregator(training_config: RecommenderFederatedTrainingConfig) -> Any:
    try:
        from lcc_lib.aggregation.secure_aggregator import (
            SecureAggregationConfig,
            SecureAggregator,
        )
        from lcc_lib.coding.field_ops import FieldConfig
        from lcc_lib.coding.share_codec import ShareEncodingConfig
        from lcc_lib.quantization.quantizer import QuantizationConfig
    except ImportError as exc:  # pragma: no cover - depends on optional sibling install
        raise ImportError(
            "Clustered recommender training requires `lcc-lib`. Install the sibling package, "
            "for example with `python3 -m pip install ../lcc-lib`."
        ) from exc

    return SecureAggregator(
        SecureAggregationConfig(
            field_config=FieldConfig(modulus=training_config.secure_field_modulus),
            quantization=QuantizationConfig(
                field_modulus=training_config.secure_field_modulus,
                scale=training_config.secure_quantization_scale,
            ),
            encoding=ShareEncodingConfig(
                num_helpers=training_config.secure_num_helpers,
                privacy_threshold=training_config.secure_privacy_threshold,
                reconstruction_threshold=training_config.secure_reconstruction_threshold,
                seed=training_config.secure_seed,
            ),
            compute_mean=False,
        )
    )


def _build_secure_clustering_components(
    training_config: RecommenderFederatedTrainingConfig,
    clustering_config: RecommenderClusteringConfig,
    seed: int,
) -> tuple[Any, Any, Any, Any]:
    try:
        from lcc_lib import ClusteringConfig, LCCConfig, QuantizationConfig, SimulationConfig
        from lcc_lib.protocols.clustering import SecureClustering
    except ImportError as exc:  # pragma: no cover - depends on optional sibling install
        raise ImportError(
            "Clustered recommender training requires `lcc-lib`. Install the sibling package, "
            "for example with `python3 -m pip install ../lcc-lib`."
        ) from exc

    lcc_config = LCCConfig(
        n_helpers=training_config.secure_num_helpers,
        privacy_threshold=training_config.secure_privacy_threshold,
        reconstruction_threshold=training_config.secure_reconstruction_threshold,
    )
    quantization_config = QuantizationConfig(
        prime=training_config.secure_field_modulus,
        precision_bits=int(np.log2(training_config.secure_quantization_scale)),
    )
    simulation_config = SimulationConfig(random_seed=int(seed))
    clustering = SecureClustering(
        config=ClusteringConfig(
            n_clusters=clustering_config.k,
            max_iterations=clustering_config.max_iterations,
            tolerance=clustering_config.tolerance,
        ),
        lcc_config=lcc_config,
        quantization_config=quantization_config,
        simulation_config=simulation_config,
    )
    return clustering, quantization_config, simulation_config, lcc_config
