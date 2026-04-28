from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fed_perso_xai.orchestration.recommender_training import train_federated_recommender
from fed_perso_xai.recommender.clustering import (
    ClientSideRandomProjector,
    RandomProjectionSpec,
    SecretSharedReducedVector,
    SecureClusterAssignments,
    build_random_projection_spec,
)
from fed_perso_xai.utils.config import (
    ArtifactPaths,
    RecommenderClusteringConfig,
    RecommenderFederatedTrainingConfig,
)

FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None
PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
LCC_AVAILABLE = importlib.util.find_spec("lcc_lib") is not None


def _paths(tmp_path: Path) -> ArtifactPaths:
    return ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )


def _prepare_recommender_run(
    tmp_path: Path,
    *,
    client_count: int = 4,
    feature_columns: tuple[str, ...] = ("metric_quality_z", "metric_stability_z"),
) -> tuple[ArtifactPaths, str, str, str]:
    paths = _paths(tmp_path)
    run_id = "unit-run"
    selection = "test__max-2__seed-9"
    persona = "lay"
    run_dir = paths.federated_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    for client_idx in range(client_count):
        client_id = f"client_{client_idx:03d}"
        client_dir = run_dir / "clients" / client_id
        context_dir = client_dir / "recommender_context" / selection
        label_dir = client_dir / "recommender_labels" / selection / persona
        context_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        candidates = pd.DataFrame(
            {
                "client_id": [client_id] * 4,
                "dataset_index": [0, 0, 1, 1],
                "instance_id": ["i0", "i0", "i1", "i1"],
                "method_variant": ["a", "b", "a", "b"],
                feature_columns[0]: [
                    1.0 + client_idx,
                    -1.0 - client_idx,
                    1.5 + client_idx,
                    -1.5 - client_idx,
                ],
                feature_columns[1]: [
                    0.2 * (client_idx + 1),
                    -0.2 * (client_idx + 1),
                    0.3 * (client_idx + 1),
                    -0.3 * (client_idx + 1),
                ],
                "candidate_index_within_instance": [0, 1, 0, 1],
            }
        )
        labels = pd.DataFrame(
            {
                "client_id": [client_id] * 2,
                "dataset_index": [0, 1],
                "pair_1": ["a", "a"],
                "pair_2": ["b", "b"],
                "label": [0, 0],
                "split": ["train", "test"],
            }
        )
        candidates.to_parquet(context_dir / "candidate_context.parquet", index=False)
        labels.to_parquet(label_dir / "pairwise_labels.parquet", index=False)
        (label_dir / "simulation_metadata.json").write_text(
            json.dumps(
                {
                    "instance_split": {
                        "train_dataset_indices": [0],
                        "test_dataset_indices": [1],
                    }
                }
            ),
            encoding="utf-8",
        )
    return paths, run_id, selection, persona


@pytest.mark.skipif(not FLOWER_AVAILABLE, reason="Flower is required for non-clustered recommender FL tests.")
@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_recommender_clustering_disabled_keeps_existing_behavior(tmp_path: Path) -> None:
    paths, run_id, selection, persona = _prepare_recommender_run(tmp_path, client_count=2)

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            paths=paths,
            rounds=2,
            epochs=5,
            batch_size=2,
            learning_rate=0.2,
            simulation_backend="debug-sequential",
            min_available_clients=2,
            top_k=(1, 2),
            clustering=RecommenderClusteringConfig(enabled=False),
        )
    )

    assert artifacts.model_artifact_path.exists()
    assert not artifacts.cluster_manifest_path.exists()
    assert metadata["clustered"] is False
    assert metadata["cluster_model_artifact_paths"] == {}


@pytest.mark.skipif(not LCC_AVAILABLE, reason="lcc-lib is required for clustered recommender tests.")
def test_client_side_projector_secret_shares_reduced_representation_only() -> None:
    config = RecommenderFederatedTrainingConfig(
        run_id="unit-run",
        selection_id="selection-0",
        persona="lay",
        clustering=RecommenderClusteringConfig(enabled=True),
    )
    projector = ClientSideRandomProjector(config)
    projection_spec = RandomProjectionSpec(
        projection_matrix=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
            dtype=np.float64,
        ),
        requested_components=2,
        seed=11,
    )
    private_vector = projector.build_private_reduced_vector(
        client_id="client_000",
        parameters=[np.asarray([1.0, 2.0], dtype=np.float64), np.asarray([0.25], dtype=np.float64)],
        projection_spec=projection_spec,
        round_id=1,
    )

    assert isinstance(private_vector, SecretSharedReducedVector)
    assert private_vector.dimension == 2
    assert not hasattr(private_vector, "reduced_vector")
    assert len(private_vector.helper_vector_shares) == config.secure_num_helpers
    assert len(private_vector.helper_squared_norm_shares) == config.secure_num_helpers
    assert all(np.asarray(share.payload).ndim == 1 for share in private_vector.helper_vector_shares)
    assert all(np.asarray(share.payload).shape == (1,) for share in private_vector.helper_squared_norm_shares)


def test_random_projection_spec_is_seeded_and_deterministic() -> None:
    spec_a = build_random_projection_spec(input_dimension=3, requested_components=8, seed=13)
    spec_b = build_random_projection_spec(input_dimension=3, requested_components=8, seed=13)
    spec_c = build_random_projection_spec(input_dimension=3, requested_components=8, seed=14)

    assert spec_a.projection_matrix.shape == (3, 8)
    assert np.allclose(spec_a.projection_matrix, spec_b.projection_matrix)
    assert not np.allclose(spec_a.projection_matrix, spec_c.projection_matrix)


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
@pytest.mark.skipif(not LCC_AVAILABLE, reason="lcc-lib is required for clustered recommender tests.")
def test_clustered_recommender_training_uses_seeded_random_projection_and_secure_cluster_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, run_id, selection, persona = _prepare_recommender_run(tmp_path, client_count=4)

    fake_local_parameters = {
        "client_000": [np.asarray([1.0, 2.0], dtype=np.float64), np.asarray([0.1], dtype=np.float64)],
        "client_001": [np.asarray([1.1, 2.1], dtype=np.float64), np.asarray([0.2], dtype=np.float64)],
        "client_002": [np.asarray([-1.0, -2.0], dtype=np.float64), np.asarray([-0.1], dtype=np.float64)],
        "client_003": [np.asarray([3.0, 3.5], dtype=np.float64), np.asarray([0.4], dtype=np.float64)],
    }
    projection_spec_calls: list[tuple[int, int, int]] = []
    captured_projector_calls: list[tuple[str, int]] = []
    secure_aggregate_calls: list[tuple[int, tuple[str, ...]]] = []

    import fed_perso_xai.fl.recommender_simulation as recommender_simulation
    import fed_perso_xai.recommender.clustering as clustering_module
    from lcc_lib.aggregation.secure_aggregator import SecureAggregator as LCCSecureAggregator

    def fake_fit_local_recommender(*, dataset, **kwargs):
        return fake_local_parameters[dataset.client_name], 0.01

    original_build = ClientSideRandomProjector.build_private_reduced_vector
    original_projection_builder = recommender_simulation.build_random_projection_spec
    def spy_projection_builder(*, input_dimension, requested_components, seed):
        projection_spec_calls.append((int(input_dimension), int(requested_components), int(seed)))
        return original_projection_builder(
            input_dimension=input_dimension,
            requested_components=requested_components,
            seed=seed,
        )

    def fail_flatten_many(self, parameter_sets):
        raise AssertionError("flatten_many must not be used in the clustering path.")

    def spy_build(self, *, client_id, parameters, projection_spec, round_id):
        captured_projector_calls.append((str(client_id), int(round_id)))
        return original_build(
            self,
            client_id=client_id,
            parameters=parameters,
            projection_spec=projection_spec,
            round_id=round_id,
        )

    def fake_cluster(self, shared_reduced_vectors, *, projection_spec, seed, clustering_config):
        assert clustering_config.k == 3
        assert clustering_config.pca_components == 8
        assert isinstance(projection_spec, RandomProjectionSpec)
        assert projection_spec.seed == 13
        assert projection_spec.projection_matrix.shape == (3, 8)
        assert all(isinstance(item, SecretSharedReducedVector) for item in shared_reduced_vectors)
        assert all(not hasattr(item, "reduced_vector") for item in shared_reduced_vectors)
        assert all(item.dimension == 8 for item in shared_reduced_vectors)
        labels = np.asarray([0, 0, 1, 2], dtype=np.int64)
        return SecureClusterAssignments(
            labels=labels,
            centroids=np.zeros((3, shared_reduced_vectors[0].dimension), dtype=np.float64),
            iterations=2,
            initial_centroid_indices=tuple(),
            secure_metadata={
                "method": clustering_config.method,
                "seed": int(seed),
                "iterations": 2,
                "n_clusters": 3,
                "helper_count": 5,
                "helper_ids": [0, 1, 2, 3, 4],
                "helper_evaluation_points": [11, 12, 13, 14, 15],
                "server_observes_raw_weights": False,
                "server_observes_reduced_vectors": False,
                "server_observes_reconstructed_distances": True,
            },
        )

    original_secure_aggregate = LCCSecureAggregator.aggregate

    def spy_secure_aggregate(self, client_vectors, round_id, client_ids=None):
        secure_aggregate_calls.append((int(round_id), tuple(client_ids or ())))
        return original_secure_aggregate(self, client_vectors, round_id, client_ids)

    monkeypatch.setattr(recommender_simulation, "_fit_local_recommender", fake_fit_local_recommender)
    monkeypatch.setattr(recommender_simulation, "build_random_projection_spec", spy_projection_builder)
    monkeypatch.setattr(clustering_module.RecommenderWeightVectorExtractor, "flatten_many", fail_flatten_many)
    monkeypatch.setattr(clustering_module.ClientSideRandomProjector, "build_private_reduced_vector", spy_build)
    monkeypatch.setattr(clustering_module.SecureKMeansClusterer, "cluster", fake_cluster)
    monkeypatch.setattr(LCCSecureAggregator, "aggregate", spy_secure_aggregate)

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            paths=paths,
            recommender_type="svm_rank",
            rounds=2,
            epochs=2,
            batch_size=2,
            learning_rate=0.1,
            seed=13,
            top_k=(1, 2),
            clustering=RecommenderClusteringConfig(enabled=True),
        )
    )

    assert metadata["clustered"] is True
    assert projection_spec_calls == [(3, 8, 13)]
    assert len(captured_projector_calls) == 8
    assert {call[0] for call in captured_projector_calls} == {
        "client_000",
        "client_001",
        "client_002",
        "client_003",
    }
    assert {call[1] for call in captured_projector_calls} == {1, 2}
    assert {call[1] for call in secure_aggregate_calls[:3]} == {
        ("client_000", "client_001"),
        ("client_002",),
        ("client_003",),
    }

    manifest = json.loads(artifacts.cluster_manifest_path.read_text(encoding="utf-8"))
    assert manifest["k"] == 3
    assert manifest["pca_components"] == 8
    assert set(manifest["final_cluster_model_checkpoint_paths"]) == {"0", "1", "2"}

    round_one = json.loads((artifacts.cluster_rounds_dir / "round_0001.json").read_text(encoding="utf-8"))
    assert round_one["assignments"] == {
        "client_000": 0,
        "client_001": 0,
        "client_002": 1,
        "client_003": 2,
    }
    assert round_one["cluster_sizes"] == {"0": 2, "1": 1, "2": 1}
    assert round_one["projection"]["projection_applied"] == "client_side"
    assert round_one["projection"]["projection_type"] == "gaussian_random"
    assert round_one["projection"]["projection_seed"] == 13
    assert round_one["projection"]["data_dependent_fit"] is False
    assert round_one["secure_clustering"]["server_observes_raw_weights"] is False
    assert round_one["secure_clustering"]["server_observes_reduced_vectors"] is False
    assert len(round_one["cluster_model_checkpoint_paths"]) == 3
    for relative_path in round_one["cluster_model_checkpoint_paths"].values():
        assert (artifacts.run_dir / relative_path).exists()

    evaluation = json.loads(artifacts.evaluation_summary_path.read_text(encoding="utf-8"))
    assert evaluation["status"] == "evaluated_clustered"
    assert len(evaluation["clusters"]) == 3


@pytest.mark.parametrize("recommender_type", ["svm_rank", "pairwise_logistic"])
@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
@pytest.mark.skipif(not LCC_AVAILABLE, reason="lcc-lib is required for clustered recommender tests.")
def test_clustered_recommender_training_supports_both_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recommender_type: str,
) -> None:
    paths, run_id, selection, persona = _prepare_recommender_run(tmp_path, client_count=3)

    import fed_perso_xai.recommender.clustering as clustering_module

    def fake_cluster(self, shared_reduced_vectors, *, projection_spec, seed, clustering_config):
        labels = np.asarray([0, 1, 2], dtype=np.int64)
        return SecureClusterAssignments(
            labels=labels,
            centroids=np.zeros((3, shared_reduced_vectors[0].dimension), dtype=np.float64),
            iterations=1,
            initial_centroid_indices=tuple(),
            secure_metadata={
                "method": clustering_config.method,
                "seed": int(seed),
                "iterations": 1,
                "n_clusters": 3,
                "helper_count": 5,
                "helper_ids": [0, 1, 2, 3, 4],
                "helper_evaluation_points": [11, 12, 13, 14, 15],
                "server_observes_raw_weights": False,
                "server_observes_reduced_vectors": False,
                "server_observes_reconstructed_distances": True,
            },
        )

    monkeypatch.setattr(clustering_module.SecureKMeansClusterer, "cluster", fake_cluster)

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            recommender_type=recommender_type,
            paths=paths,
            rounds=2,
            epochs=3,
            batch_size=2,
            learning_rate=0.2,
            seed=7,
            top_k=(1, 2),
            clustering=RecommenderClusteringConfig(enabled=True),
        )
    )

    assert artifacts.cluster_manifest_path.exists()
    assert metadata["status"] == "completed"
    assert metadata["clustered"] is True
    assert metadata["recommender_type"] == recommender_type


def test_recommender_clustering_config_defaults_and_validation() -> None:
    config = RecommenderClusteringConfig(enabled=True)
    assert config.method == "secure_kmeans"
    assert config.k == 3
    assert config.pca_components == 8

    with pytest.raises(ValueError, match="Unsupported clustering.method"):
        RecommenderClusteringConfig(enabled=True, method="missing")
