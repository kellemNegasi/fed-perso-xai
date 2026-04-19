from __future__ import annotations

import json

import numpy as np

from fed_perso_xai.data.partitioning import ClientSplit
from fed_perso_xai.data.serialization import save_federated_dataset
from fed_perso_xai.orchestration.training import train_from_saved_partitions
from fed_perso_xai.utils.config import FederatedTrainingConfig


def test_federated_training_smoke(tmp_path) -> None:
    rng = np.random.default_rng(0)
    client_splits = []
    for client_id in range(3):
        X_train = rng.normal(size=(16, 4))
        logits_train = X_train[:, 0] - 0.5 * X_train[:, 1]
        y_train = (logits_train > 0.0).astype(np.int64)
        X_test = rng.normal(size=(8, 4))
        logits_test = X_test[:, 0] - 0.5 * X_test[:, 1]
        y_test = (logits_test > 0.0).astype(np.int64)
        client_splits.append(
            ClientSplit(
                client_id=client_id,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        )

    save_federated_dataset(
        dataset_name="adult_income",
        output_root=tmp_path / "datasets",
        num_clients=3,
        alpha=1.0,
        seed=42,
        feature_names=["f0", "f1", "f2", "f3"],
        preprocessing_info={"feature_names": ["f0", "f1", "f2", "f3"]},
        client_splits=client_splits,
    )
    config = FederatedTrainingConfig(
        dataset_name="adult_income",
        data_root=tmp_path / "datasets",
        results_root=tmp_path / "results",
        num_clients=3,
        alpha=1.0,
        rounds=3,
        local_epochs=2,
        batch_size=8,
        learning_rate=0.1,
        fit_fraction=1.0,
        evaluate_fraction=1.0,
        min_available_clients=2,
    )
    artifacts, summary = train_from_saved_partitions(config)
    assert artifacts.metrics_path.exists()
    assert artifacts.model_path.exists()
    assert "aggregated_weighted" in summary
    assert "aggregated_pooled" in summary
    assert "threshold_analysis" in summary
    assert "best_by_f1" in summary["threshold_analysis"]
    assert len(summary["per_client"]) == 3

    persisted = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert persisted["simulation_backend"] == "sequential_flower_fallback"
    assert "aggregated_pooled" in persisted
