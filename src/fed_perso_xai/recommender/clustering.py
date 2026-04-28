"""Helpers for clustered federated recommender training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from fed_perso_xai.utils.config import RecommenderClusteringConfig, RecommenderFederatedTrainingConfig


@dataclass(frozen=True)
class RandomProjectionSpec:
    """Global stateless random projection broadcast to clients for local projection."""

    projection_matrix: np.ndarray
    requested_components: int
    seed: int

    def transform(self, flat_vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(flat_vector, dtype=np.float64).reshape(-1)
        if vector.shape[0] != self.projection_matrix.shape[0]:
            raise ValueError(
                f"Expected flattened vector length {self.projection_matrix.shape[0]}, got {vector.shape[0]}."
            )
        return vector @ self.projection_matrix

    @property
    def input_dimension(self) -> int:
        return int(self.projection_matrix.shape[0])

    @property
    def actual_components(self) -> int:
        return int(self.projection_matrix.shape[1])

    def to_metadata(self) -> dict[str, Any]:
        return {
            "requested_components": int(self.requested_components),
            "actual_components": int(self.actual_components),
            "input_dimension": int(self.input_dimension),
            "projection_matrix_shape": [int(value) for value in self.projection_matrix.shape],
            "projection_type": "gaussian_random",
            "projection_seed": int(self.seed),
            "data_dependent_fit": False,
            "projection_applied": "client_side",
            "projection_basis_scope": "global_seeded_matrix",
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


@dataclass(frozen=True)
class SecretSharedReducedVector:
    """Opaque client-side reduced representation exposed to the server as shares only."""

    client_id: str
    helper_vector_shares: tuple[Any, ...]
    helper_squared_norm_shares: tuple[Any, ...]
    dimension: int


@dataclass(frozen=True)
class _PrivateClusteringProtocol:
    field_config: Any
    vector_quantizer: Any
    distance_quantizer: Any
    encoding_config: Any
    reconstructor: Any
    helper_runtime_class: Any
    helper_ids: tuple[int, ...]
    helper_evaluation_points: tuple[int, ...]
    vector_scale: int
    distance_scale: int


@dataclass(frozen=True)
class _DerivedHelperPayload:
    helper_id: int
    evaluation_point: int
    payload: np.ndarray


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


def build_random_projection_spec(
    *,
    input_dimension: int,
    requested_components: int,
    seed: int,
) -> RandomProjectionSpec:
    if int(input_dimension) <= 0:
        raise ValueError("input_dimension must be positive.")
    if int(requested_components) <= 0:
        raise ValueError("requested_components must be positive.")
    rng = np.random.default_rng(int(seed))
    scale = 1.0 / math.sqrt(float(requested_components))
    projection_matrix = (
        rng.standard_normal(size=(int(input_dimension), int(requested_components))).astype(np.float64) * scale
    )
    return RandomProjectionSpec(
        projection_matrix=projection_matrix,
        requested_components=int(requested_components),
        seed=int(seed),
    )


class ClientSideRandomProjector:
    """Client-local flatten -> random projection -> secret sharing for clustered training."""

    def __init__(self, training_config: RecommenderFederatedTrainingConfig) -> None:
        self.training_config = training_config
        self.extractor = RecommenderWeightVectorExtractor()

    def build_private_reduced_vector(
        self,
        *,
        client_id: str,
        parameters: Sequence[np.ndarray],
        projection_spec: RandomProjectionSpec,
        round_id: int,
    ) -> SecretSharedReducedVector:
        protocol = _build_private_clustering_protocol(self.training_config)
        share_encoder = _build_share_encoder(protocol, client_id=client_id, round_id=round_id)
        flat_vector = self.extractor.flatten(parameters)
        reduced_vector = projection_spec.transform(flat_vector)
        squared_norm = np.asarray([float(np.dot(reduced_vector, reduced_vector))], dtype=np.float64)
        vector_shares = tuple(
            share_encoder.encode(protocol.vector_quantizer.quantize(reduced_vector))
        )
        squared_norm_shares = tuple(
            share_encoder.encode(protocol.distance_quantizer.quantize(squared_norm))
        )
        return SecretSharedReducedVector(
            client_id=str(client_id),
            helper_vector_shares=vector_shares,
            helper_squared_norm_shares=squared_norm_shares,
            dimension=int(reduced_vector.shape[0]),
        )


class SecureKMeansClusterer:
    """Secure K-means over secret-shared client-side projections."""

    def __init__(self, training_config: RecommenderFederatedTrainingConfig) -> None:
        self.training_config = training_config

    def cluster(
        self,
        shared_reduced_vectors: Sequence[SecretSharedReducedVector],
        *,
        projection_spec: RandomProjectionSpec,
        seed: int,
        clustering_config: RecommenderClusteringConfig,
    ) -> SecureClusterAssignments:
        if not shared_reduced_vectors:
            raise ValueError("shared_reduced_vectors must not be empty.")
        if len(shared_reduced_vectors) < clustering_config.k:
            raise ValueError("clustering.k cannot exceed the number of participating clients.")

        protocol = _build_private_clustering_protocol(self.training_config)
        dimension = int(shared_reduced_vectors[0].dimension)
        current = _initialize_centroids(
            projection_spec=projection_spec,
            dimension=dimension,
            n_clusters=clustering_config.k,
            seed=seed,
        )
        labels = np.zeros(len(shared_reduced_vectors), dtype=np.int64)
        last_distance_helper_ids: tuple[int, ...] = ()
        last_distance_evaluation_points: tuple[int, ...] = ()
        for iteration in range(1, clustering_config.max_iterations + 1):
            distance_matrix, helper_ids, evaluation_points = self._reconstruct_distances(
                shared_reduced_vectors,
                current,
                protocol,
            )
            labels = np.argmin(distance_matrix, axis=1).astype(np.int64, copy=False)
            updated = self._recompute_centroids(
                shared_reduced_vectors,
                labels,
                current,
                protocol,
                clustering_config.k,
                round_seed=seed + iteration,
            )
            shift = float(np.linalg.norm(updated - current))
            current = updated
            last_distance_helper_ids = helper_ids
            last_distance_evaluation_points = evaluation_points
            if shift <= clustering_config.tolerance:
                break

        distance_matrix, helper_ids, evaluation_points = self._reconstruct_distances(
            shared_reduced_vectors,
            current,
            protocol,
        )
        labels = np.argmin(distance_matrix, axis=1).astype(np.int64, copy=False)
        secure_metadata = {
            "method": clustering_config.method,
            "seed": int(seed),
            "iterations": int(iteration),
            "n_clusters": int(clustering_config.k),
            "max_iterations": int(clustering_config.max_iterations),
            "tolerance": float(clustering_config.tolerance),
            "helper_count": int(len(protocol.helper_ids)),
            "privacy_threshold": int(protocol.encoding_config.privacy_threshold),
            "reconstruction_threshold": int(protocol.encoding_config.resolved_reconstruction_threshold),
            "field_modulus": int(protocol.field_config.modulus),
            "vector_quantization_scale": int(protocol.vector_scale),
            "distance_quantization_scale": int(protocol.distance_scale),
            "helper_ids": [int(value) for value in helper_ids or last_distance_helper_ids],
            "helper_evaluation_points": [
                int(value) for value in evaluation_points or last_distance_evaluation_points
            ],
            "server_observes_raw_weights": False,
            "server_observes_reduced_vectors": False,
            "server_observes_reconstructed_distances": True,
            "projection_applied": "client_side",
        }
        return SecureClusterAssignments(
            labels=labels,
            centroids=current.astype(np.float64, copy=True),
            iterations=int(iteration),
            initial_centroid_indices=tuple(),
            secure_metadata=secure_metadata,
        )

    def _reconstruct_distances(
        self,
        shared_reduced_vectors: Sequence[SecretSharedReducedVector],
        centroids: np.ndarray,
        protocol: _PrivateClusteringProtocol,
    ) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
        centroid_array = np.asarray(centroids, dtype=np.float64)
        centroid_quantized = protocol.vector_quantizer.quantize(centroid_array.reshape(-1)).reshape(
            centroid_array.shape
        )
        centroid_norms = np.sum(centroid_array**2, axis=1)
        distances = np.zeros((len(shared_reduced_vectors), centroid_array.shape[0]), dtype=np.float64)
        helper_ids = protocol.helper_ids
        evaluation_points = protocol.helper_evaluation_points
        for row_index, shared_vector in enumerate(shared_reduced_vectors):
            if shared_vector.dimension != centroid_array.shape[1]:
                raise ValueError("Shared reduced vector dimension does not match centroid dimension.")
            norm_share_lookup = {
                int(share.helper_id): share for share in shared_vector.helper_squared_norm_shares
            }
            helper_payloads: list[_DerivedHelperPayload] = []
            for vector_share in shared_vector.helper_vector_shares:
                helper_id = int(vector_share.helper_id)
                norm_share = norm_share_lookup[helper_id]
                vector_payload = np.asarray(vector_share.payload, dtype=np.int64).reshape(-1)
                dot_terms = np.mod(centroid_quantized @ vector_payload, protocol.field_config.modulus).astype(
                    np.int64,
                    copy=False,
                )
                partial = np.mod(
                    int(np.asarray(norm_share.payload, dtype=np.int64).reshape(-1)[0]) - (2 * dot_terms),
                    protocol.field_config.modulus,
                ).astype(np.int64, copy=False)
                helper_payloads.append(
                    _DerivedHelperPayload(
                        helper_id=helper_id,
                        evaluation_point=int(vector_share.evaluation_point),
                        payload=partial,
                    )
                )
            reconstructed = protocol.reconstructor.reconstruct(helper_payloads)
            distances[row_index] = protocol.distance_quantizer.dequantize(reconstructed) + centroid_norms
        return distances, helper_ids, evaluation_points

    def _recompute_centroids(
        self,
        shared_reduced_vectors: Sequence[SecretSharedReducedVector],
        labels: np.ndarray,
        previous_centroids: np.ndarray,
        protocol: _PrivateClusteringProtocol,
        n_clusters: int,
        round_seed: int,
    ) -> np.ndarray:
        updated = np.asarray(previous_centroids, dtype=np.float64).copy()
        for cluster_id in range(n_clusters):
            members = [
                shared_vector
                for shared_vector, label in zip(shared_reduced_vectors, labels, strict=True)
                if int(label) == cluster_id
            ]
            if not members:
                continue
            runtime = protocol.helper_runtime_class(
                field_config=protocol.field_config,
                helper_evaluation_points=protocol.helper_evaluation_points,
            )
            round_id = (int(round_seed) * 1_000) + int(cluster_id)
            runtime.start_round(round_id)
            try:
                for shared_vector in members:
                    for share in shared_vector.helper_vector_shares:
                        runtime.upload_share(
                            round_id=round_id,
                            client_id=shared_vector.client_id,
                            helper_id=int(share.helper_id),
                            share=np.asarray(share.payload, dtype=np.int64),
                        )
                helper_payloads = runtime.finalize_round(round_id)
            finally:
                runtime.reset_round(round_id)
            reconstructed_sum = protocol.reconstructor.reconstruct(helper_payloads)
            cluster_sum = protocol.vector_quantizer.dequantize(reconstructed_sum)
            updated[cluster_id] = cluster_sum / float(len(members))
        return updated


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
        min_contributors = 2
        for cluster_id in range(cluster_count):
            member_ids = [
                client_id
                for client_id, assigned_cluster in assignments.items()
                if int(assigned_cluster) == cluster_id
            ]
            if len(member_ids) < min_contributors:
                fallback = fallback_parameters.get(cluster_id)
                if fallback is None:
                    raise ValueError(f"Missing fallback parameters for underpopulated cluster {cluster_id}.")
                mode = (
                    "carry_forward_empty_cluster"
                    if not member_ids
                    else "carry_forward_underpopulated_cluster"
                )
                results[cluster_id] = ClusterAggregationResult(
                    parameters=[np.asarray(parameter, dtype=np.float64).copy() for parameter in fallback],
                    metadata={
                        "cluster_id": int(cluster_id),
                        "round_id": int(round_id),
                        "mode": mode,
                        "num_contributors": int(len(member_ids)),
                        "client_ids": list(member_ids),
                        "min_required_contributors": int(min_contributors),
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


def _initialize_centroids(
    *,
    projection_spec: RandomProjectionSpec,
    dimension: int,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = rng.normal(loc=0.0, scale=1e-3, size=(n_clusters, dimension)).astype(np.float64)
    if dimension == 0:
        raise ValueError("dimension must be positive.")
    scales = np.linalg.norm(projection_spec.projection_matrix, axis=0)
    if scales.shape[0] < dimension:
        scales = np.pad(scales, (0, dimension - scales.shape[0]), constant_values=1.0)
    scales = np.maximum(scales[:dimension], 1e-3)
    for cluster_id in range(n_clusters):
        axis = cluster_id % dimension
        sign = -1.0 if cluster_id % 2 else 1.0
        centroids[cluster_id, axis] += sign * float(scales[axis])
    return centroids


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


def _build_private_clustering_protocol(
    training_config: RecommenderFederatedTrainingConfig,
) -> _PrivateClusteringProtocol:
    try:
        from lcc_lib.coding.field_ops import FieldConfig
        from lcc_lib.coding.share_codec import ShareEncoder, ShareEncodingConfig, ShareReconstructor
        from lcc_lib.quantization.quantizer import QuantizationConfig, Quantizer
        from lcc_lib.runtime.in_memory_helpers import InMemoryHelperRuntime
    except ImportError as exc:  # pragma: no cover - depends on optional sibling install
        raise ImportError(
            "Clustered recommender training requires `lcc-lib`. Install the sibling package, "
            "for example with `python3 -m pip install ../lcc-lib`."
        ) from exc

    vector_scale = max(1, math.isqrt(int(training_config.secure_quantization_scale)))
    distance_scale = max(1, vector_scale * vector_scale)
    field_config = FieldConfig(modulus=training_config.secure_field_modulus)
    encoding_config = ShareEncodingConfig(
        num_helpers=training_config.secure_num_helpers,
        privacy_threshold=training_config.secure_privacy_threshold,
        reconstruction_threshold=training_config.secure_reconstruction_threshold,
        seed=training_config.secure_seed,
    )
    return _PrivateClusteringProtocol(
        field_config=field_config,
        vector_quantizer=Quantizer(
            QuantizationConfig(
                field_modulus=training_config.secure_field_modulus,
                scale=vector_scale,
            )
        ),
        distance_quantizer=Quantizer(
            QuantizationConfig(
                field_modulus=training_config.secure_field_modulus,
                scale=distance_scale,
            )
        ),
        encoding_config=encoding_config,
        reconstructor=ShareReconstructor(field_config, encoding_config),
        helper_runtime_class=InMemoryHelperRuntime,
        helper_ids=tuple(range(encoding_config.num_helpers)),
        helper_evaluation_points=encoding_config.helper_evaluation_points,
        vector_scale=int(vector_scale),
        distance_scale=int(distance_scale),
    )


def _build_share_encoder(
    protocol: _PrivateClusteringProtocol,
    *,
    client_id: str,
    round_id: int,
) -> Any:
    from lcc_lib.coding.share_codec import ShareEncoder, ShareEncodingConfig

    stable_hash = sum(ord(character) for character in str(client_id))
    encoder_config = ShareEncodingConfig(
        num_helpers=protocol.encoding_config.num_helpers,
        privacy_threshold=protocol.encoding_config.privacy_threshold,
        reconstruction_threshold=protocol.encoding_config.reconstruction_threshold,
        secret_point=protocol.encoding_config.secret_point,
        helper_point_start=protocol.encoding_config.helper_point_start,
        seed=int(protocol.encoding_config.seed + stable_hash + (round_id * 10_000)),
    )
    return ShareEncoder(protocol.field_config, encoder_config)
