from __future__ import annotations

import json

from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.utils.config import ArtifactPaths, DataPreparationConfig, PartitionConfig


def test_prepare_data_writes_shared_and_client_artifacts(mock_openml, tmp_path) -> None:
    mock_openml("adult_income")
    config = DataPreparationConfig(
        dataset_name="adult_income",
        seed=11,
        paths=ArtifactPaths(
            prepared_root=tmp_path / "prepared",
            partition_root=tmp_path / "datasets",
            centralized_root=tmp_path / "centralized",
            federated_root=tmp_path / "federated",
            comparison_root=tmp_path / "comparisons",
            cache_dir=tmp_path / "cache",
        ),
        partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
    )
    result = prepare_federated_dataset(config)

    prepared_root = result.prepared_artifacts.root_dir
    assert (prepared_root / "global_train.npz").exists()
    assert (prepared_root / "global_eval.npz").exists()
    assert (prepared_root / "pooled_client_test.npz").exists()
    assert (prepared_root / "preprocessor.joblib").exists()
    assert result.prepared_artifacts.split_metadata_path.exists()

    feature_metadata = json.loads(result.prepared_artifacts.feature_metadata_path.read_text(encoding="utf-8"))
    assert feature_metadata["schema_version"] == "stage1_feature_metadata_v2"
    assert "feature_lineage" in feature_metadata
    split_metadata = json.loads(result.prepared_artifacts.split_metadata_path.read_text(encoding="utf-8"))
    assert split_metadata["global_eval"]["transform_diagnostics"]["split_name"] == "global_eval"

    partition_root = result.federated_artifacts.root_dir
    assert (partition_root / "client_0" / "train.npz").exists()
    assert (partition_root / "client_0" / "test.npz").exists()
    metadata = json.loads(result.federated_artifacts.partition_metadata_path.read_text(encoding="utf-8"))
    assert metadata["prepared_root"] == str(prepared_root)
    assert metadata["clients"][0]["train_size"] > 0
