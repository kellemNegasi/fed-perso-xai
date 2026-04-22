from __future__ import annotations

import importlib.util
import time

import numpy as np
import pytest

from fed_perso_xai.data.serialization import load_client_datasets
from fed_perso_xai.models import load_global_model
from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.orchestration.stage_b_training import train_federated_stage_b
from fed_perso_xai.utils.config import (
    ArtifactPaths,
    DataPreparationConfig,
    FederatedTrainingConfig,
    LogisticRegressionConfig,
    PartitionConfig,
)
from fed_perso_xai.utils.paths import partition_root

FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None


def _build_paths(tmp_path):
    return ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )


@pytest.mark.skipif(
    not FLOWER_AVAILABLE,
    reason="Flower is not installed; Stage B federated training tests require Flower.",
)
def test_stage_b_loads_persisted_partitions_and_writes_artifacts(mock_openml, tmp_path) -> None:
    mock_openml("adult_income")
    paths = _build_paths(tmp_path)
    prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="adult_income",
            seed=12,
            paths=paths,
            partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
        )
    )

    explicit_partition_root = partition_root(
        paths.partition_root,
        "adult_income",
        3,
        1.0,
        12,
    )
    artifacts, metadata = train_federated_stage_b(
        FederatedTrainingConfig(
            dataset_name="adult_income",
            seed=12,
            paths=paths,
            model=LogisticRegressionConfig(epochs=2, batch_size=4, learning_rate=0.1),
            num_clients=3,
            alpha=1.0,
            rounds=2,
            simulation_backend="debug-sequential",
        ),
        run_id="stage-b-explicit-run",
        partition_data_root=explicit_partition_root,
    )

    assert artifacts.model_artifact_path.exists()
    assert artifacts.model_metadata_path.exists()
    assert artifacts.training_metadata_path.exists()
    assert artifacts.training_history_path.exists()
    assert artifacts.completion_marker_path.exists()
    assert metadata["partition_data_root"] == str(explicit_partition_root)
    assert metadata["status"] == "completed"
    assert metadata["run_id"] == "stage-b-explicit-run"

    loaded = load_global_model(artifacts.run_dir)
    client = load_client_datasets(explicit_partition_root, 3)[0]
    probabilities = loaded.model.predict_proba(client.test.X)
    assert probabilities.shape[0] == client.test.X.shape[0]
    assert loaded.metadata["model_artifact_path"] == "model/global_model.npz"
    assert loaded.metadata["training_metadata_path"] == "training/training_metadata.json"


@pytest.mark.skipif(
    not FLOWER_AVAILABLE,
    reason="Flower is not installed; Stage B federated training tests require Flower.",
)
def test_stage_b_skip_existing_and_force_rerun(mock_openml, tmp_path) -> None:
    mock_openml("adult_income")
    paths = _build_paths(tmp_path)
    prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="adult_income",
            seed=18,
            paths=paths,
            partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
        )
    )
    config = FederatedTrainingConfig(
        dataset_name="adult_income",
        seed=18,
        paths=paths,
        model=LogisticRegressionConfig(epochs=2, batch_size=4, learning_rate=0.1),
        num_clients=3,
        alpha=1.0,
        rounds=2,
        simulation_backend="debug-sequential",
    )

    artifacts, initial_metadata = train_federated_stage_b(config)
    initial_text = artifacts.training_metadata_path.read_text(encoding="utf-8")
    assert initial_metadata["status"] == "completed"

    _, skipped_metadata = train_federated_stage_b(config)
    assert skipped_metadata["status"] == "skipped_existing"
    assert skipped_metadata["skipped"] is True
    assert artifacts.training_metadata_path.read_text(encoding="utf-8") == initial_text

    time.sleep(0.01)
    _, forced_metadata = train_federated_stage_b(config, force=True)
    forced_text = artifacts.training_metadata_path.read_text(encoding="utf-8")
    assert forced_metadata["status"] == "completed"
    assert forced_metadata["force_requested"] is True
    assert forced_text != initial_text

    loaded = load_global_model(artifacts.run_dir)
    parameters = loaded.model.get_parameters()
    assert all(np.asarray(parameter).size > 0 for parameter in parameters)
