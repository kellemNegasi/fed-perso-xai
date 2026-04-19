from __future__ import annotations

import importlib.util
import json

import pytest

from fed_perso_xai.evaluation.comparison import build_baseline_comparison
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

FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None
RAY_AVAILABLE = importlib.util.find_spec("ray") is not None


def _build_paths(tmp_path):
    return ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )


def test_centralized_run_manifest_creation(mock_openml, tmp_path) -> None:
    mock_openml("adult_income")
    paths = _build_paths(tmp_path)
    prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="adult_income",
            seed=17,
            paths=paths,
            partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
        )
    )

    result_dir, summary = train_centralized_from_prepared(
        CentralizedTrainingConfig(
            dataset_name="adult_income",
            seed=17,
            paths=paths,
            model=LogisticRegressionConfig(epochs=2, batch_size=4, learning_rate=0.1),
        )
    )

    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == summary["run_id"]
    assert manifest["mode"] == "centralized"
    assert manifest["dataset_name"] == "adult_income"
    assert manifest["timestamp"]
    assert "git_commit_hash" in manifest
    assert manifest["seed_values"] == {"global_seed": 17}
    assert manifest["artifacts"]["config_snapshot"] == "config_snapshot.json"
    assert manifest["artifacts"]["model_artifact"] == "model_parameters.npz"
    assert manifest["artifacts"]["preprocessor_artifact"] == "preprocessor.joblib"
    assert manifest["artifacts"]["feature_metadata_artifact"] == "feature_metadata.json"
    assert manifest["artifacts"]["metrics_artifact"] == "metrics_summary.json"
    assert manifest["artifacts"]["predictions_artifacts"]["predictions_global_eval"] == "predictions_global_eval.npz"
    assert manifest["artifacts"]["metadata_artifacts"]["split_metadata"] == "split_metadata.json"


def test_centralized_and_federated_training_outputs_are_compatible(mock_openml, tmp_path) -> None:
    if not FLOWER_AVAILABLE:
        pytest.skip("Flower is not installed; skipping federated training contract test.")
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
    assert "predictions_global_eval" in centralized_manifest["artifacts"]["predictions_artifacts"]
    assert "runtime_report" in federated_manifest["artifacts"]["additional_artifacts"]


@pytest.mark.skipif(
    not FLOWER_AVAILABLE or not RAY_AVAILABLE,
    reason="Flower with Ray support is not installed; real Flower simulation smoke test requires the ray extra.",
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


def test_compare_baselines_reporting() -> None:
    report = build_baseline_comparison(
        centralized_summary={
            "run_id": "centralized-adult_income-seed33",
            "dataset_name": "adult_income",
            "result_dir": "centralized/adult_income/seed_33",
            "run_manifest_path": "centralized/adult_income/seed_33/run_manifest.json",
            "evaluation": {
                "global_eval": {
                    "split_name": "global_eval",
                    "class_balance": {"num_examples": 12},
                    "probability_summary": {"mean": 0.5},
                    "metrics": {"loss": 0.3, "accuracy": 0.8, "roc_auc": 0.9},
                },
                "pooled_client_test": {
                    "split_name": "pooled_client_test",
                    "class_balance": {"num_examples": 9},
                    "probability_summary": {"mean": 0.55},
                    "metrics": {"loss": 0.35, "accuracy": 0.78},
                },
            },
        },
        federated_summary={
            "run_id": "federated-adult_income-3clients-alpha1.0-seed33",
            "dataset_name": "adult_income",
            "result_dir": "federated/adult_income/3_clients/alpha_1.0/seed_33",
            "run_manifest_path": "federated/adult_income/3_clients/alpha_1.0/seed_33/run_manifest.json",
            "runtime": {"actual_backend": "debug-sequential"},
            "evaluation": {
                "client_test_weighted": {
                    "split_name": "client_test_weighted",
                    "class_balance": {"num_examples": 9},
                    "probability_summary": {"mean": 0.51},
                    "metrics": {"loss": 0.4, "accuracy": 0.75},
                },
                "client_test_pooled": {
                    "split_name": "client_test",
                    "class_balance": {"num_examples": 9},
                    "probability_summary": {"mean": 0.52},
                    "metrics": {"loss": 0.38, "accuracy": 0.77, "roc_auc": 0.88},
                },
                "per_client": [
                    {"client_id": 0, "num_examples": 3, "metrics": {"loss": 0.5, "accuracy": 0.7}},
                    {"client_id": 1, "num_examples": 3, "metrics": {"loss": 0.4, "accuracy": 0.8}},
                    {"client_id": 2, "num_examples": 3, "metrics": {"loss": 0.3, "accuracy": 0.75}},
                ],
            },
        },
        centralized_manifest={
            "run_id": "centralized-adult_income-seed33",
            "mode": "centralized",
        },
        federated_manifest={
            "run_id": "federated-adult_income-3clients-alpha1.0-seed33",
            "mode": "federated",
        },
    )
    assert report["predictive_metric_comparison"][0]["metric"] == "loss"
    assert "split_reports" in report
    assert "class_balance" in report["split_reports"]["centralized_global_eval"]
    assert report["federated_per_client_summary"]["num_clients"] == 3
    accuracy_row = next(
        row for row in report["predictive_metric_comparison"] if row["metric"] == "accuracy"
    )
    assert accuracy_row["absolute_difference_weighted_vs_centralized"] == pytest.approx(0.05)
    roc_auc_row = next(row for row in report["predictive_metric_comparison"] if row["metric"] == "roc_auc")
    assert roc_auc_row["metric_availability_notes"] == [
        "Metric 'roc_auc' is unavailable for centralized_pooled_client_test.",
        "Metric 'roc_auc' is unavailable for federated_client_test_weighted.",
    ]
    assert report["source_references"]["centralized"]["run_manifest_path"].endswith("run_manifest.json")


def test_compare_baselines_end_to_end(mock_openml, tmp_path) -> None:
    if not FLOWER_AVAILABLE:
        pytest.skip("Flower is not installed; skipping end-to-end comparison test.")
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
    assert persisted["source_references"]["federated"]["run_manifest_path"].endswith("run_manifest.json")
