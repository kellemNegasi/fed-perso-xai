from __future__ import annotations

import importlib.util
import json

import pytest

from fed_perso_xai.evaluation.predictions import load_prediction_artifact
from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.orchestration.training import (
    compare_centralized_and_federated,
    train_centralized_from_prepared,
    train_federated_from_prepared,
)
from fed_perso_xai.utils.config import (
    ArtifactPaths,
    CentralizedTrainingConfig,
    ComparisonConfig,
    DataPreparationConfig,
    FederatedTrainingConfig,
    LogisticRegressionConfig,
    PartitionConfig,
)


def _build_paths(tmp_path):
    return ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )


def test_centralized_and_federated_training_outputs_are_compatible(mock_openml, tmp_path) -> None:
    mock_openml("adult_income")
    paths = _build_paths(tmp_path)
    prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="adult_income",
            seed=21,
            paths=paths,
            partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
        )
    )

    _, centralized_summary = train_centralized_from_prepared(
        CentralizedTrainingConfig(
            dataset_name="adult_income",
            seed=21,
            paths=paths,
            model=LogisticRegressionConfig(epochs=3, batch_size=4, learning_rate=0.1),
        )
    )
    federated_artifacts, federated_summary = train_federated_from_prepared(
        FederatedTrainingConfig(
            dataset_name="adult_income",
            seed=21,
            paths=paths,
            model=LogisticRegressionConfig(epochs=2, batch_size=4, learning_rate=0.1),
            num_clients=3,
            alpha=1.0,
            rounds=2,
            simulation_backend="debug-sequential",
        )
    )

    centralized_predictions = load_prediction_artifact(
        paths.centralized_root / "adult_income" / "seed_21" / "predictions_global_eval.npz"
    )
    federated_predictions = load_prediction_artifact(federated_artifacts.predictions_path)
    assert centralized_predictions.dataset_name == "adult_income"
    assert federated_predictions.run_id == federated_summary["run_id"]
    assert federated_predictions.client_ids.shape[0] == federated_predictions.y_true.shape[0]
    assert centralized_summary["evaluation"]["global_eval"]["metrics"]["accuracy"] >= 0.0
    assert federated_summary["evaluation"]["client_test_pooled"]["metrics"]["accuracy"] >= 0.0

    centralized_manifest = json.loads(
        (paths.centralized_root / "adult_income" / "seed_21" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    federated_manifest = json.loads(
        (
            paths.federated_root
            / "adult_income"
            / "3_clients"
            / "alpha_1.0"
            / "seed_21"
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert "predictions_global_eval" in centralized_manifest["artifacts"]
    assert "runtime_report" in federated_manifest["artifacts"]


@pytest.mark.skipif(
    importlib.util.find_spec("ray") is None,
    reason="Ray is not installed; real Flower simulation smoke test requires the ray extra.",
)
def test_real_flower_simulation_smoke_path(mock_openml, tmp_path) -> None:
    mock_openml("bank_marketing")
    paths = _build_paths(tmp_path)
    prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="bank_marketing",
            seed=9,
            paths=paths,
            partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
        )
    )

    artifacts, summary = train_federated_from_prepared(
        FederatedTrainingConfig(
            dataset_name="bank_marketing",
            seed=9,
            paths=paths,
            model=LogisticRegressionConfig(epochs=1, batch_size=4, learning_rate=0.1),
            num_clients=3,
            alpha=1.0,
            rounds=1,
            simulation_backend="ray",
        )
    )
    assert artifacts.metrics_path.exists()
    assert summary["runtime"]["actual_backend"] == "ray"
    assert summary["round_history"][0]["round"] == 1


def test_compare_baselines_reporting(mock_openml, tmp_path) -> None:
    mock_openml("adult_income")
    paths = _build_paths(tmp_path)
    prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="adult_income",
            seed=33,
            paths=paths,
            partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
        )
    )
    train_centralized_from_prepared(
        CentralizedTrainingConfig(
            dataset_name="adult_income",
            seed=33,
            paths=paths,
            model=LogisticRegressionConfig(epochs=2, batch_size=4, learning_rate=0.1),
        )
    )
    train_federated_from_prepared(
        FederatedTrainingConfig(
            dataset_name="adult_income",
            seed=33,
            paths=paths,
            model=LogisticRegressionConfig(epochs=1, batch_size=4, learning_rate=0.1),
            num_clients=3,
            alpha=1.0,
            rounds=1,
            simulation_backend="debug-sequential",
        )
    )
    report_path, report = compare_centralized_and_federated(
        ComparisonConfig(
            dataset_name="adult_income",
            seed=33,
            num_clients=3,
            alpha=1.0,
            paths=paths,
        )
    )
    assert report_path.exists()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["predictive_metric_comparison"][0]["metric"] == "loss"
    assert "split_reports" in persisted
    assert "class_balance" in persisted["split_reports"]["centralized_global_eval"]
    assert persisted["federated_per_client_summary"]["num_clients"] == 3
