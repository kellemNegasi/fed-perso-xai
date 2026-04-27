from __future__ import annotations

import pytest

from fed_perso_xai.utils.config import (
    ComparisonConfig,
    DataPreparationConfig,
    FederatedTrainingConfig,
    LogisticRegressionConfig,
    PartitionConfig,
    PreprocessingConfig,
    RecommenderFederatedTrainingConfig,
)


def test_preprocessing_config_rejects_invalid_split_sizes() -> None:
    with pytest.raises(ValueError, match="global_eval_size"):
        PreprocessingConfig(global_eval_size=1.0)
    with pytest.raises(ValueError, match="client_test_size"):
        PreprocessingConfig(client_test_size=0.0)
    with pytest.raises(ValueError, match="fitting_mode"):
        PreprocessingConfig(fitting_mode="client_local")


def test_partition_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_clients"):
        PartitionConfig(num_clients=1, alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        PartitionConfig(num_clients=2, alpha=0.0)


def test_logistic_regression_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="epochs"):
        LogisticRegressionConfig(epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        LogisticRegressionConfig(learning_rate=0.0)


def test_federated_training_config_rejects_unknown_model_and_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        FederatedTrainingConfig(dataset_name="adult_income", model_name="missing_model")
    with pytest.raises(ValueError, match="Unsupported strategy"):
        FederatedTrainingConfig(dataset_name="adult_income", strategy_name="missing_strategy")


def test_federated_training_config_rejects_invalid_secure_aggregation_values() -> None:
    with pytest.raises(ValueError, match="secure_num_helpers"):
        FederatedTrainingConfig(dataset_name="adult_income", secure_num_helpers=0)
    with pytest.raises(ValueError, match="secure_num_helpers"):
        FederatedTrainingConfig(
            dataset_name="adult_income",
            secure_num_helpers=2,
            secure_privacy_threshold=2,
        )
    with pytest.raises(ValueError, match="secure_reconstruction_threshold"):
        FederatedTrainingConfig(
            dataset_name="adult_income",
            secure_reconstruction_threshold=2,
            secure_privacy_threshold=2,
        )


def test_recommender_training_config_rejects_unknown_recommender_type() -> None:
    with pytest.raises(ValueError, match="Unsupported recommender_type"):
        RecommenderFederatedTrainingConfig(
            run_id="unit-run",
            selection_id="selection-0",
            persona="lay",
            recommender_type="missing",
        )


def test_top_level_configs_reject_invalid_common_values() -> None:
    with pytest.raises(ValueError, match="dataset_name"):
        DataPreparationConfig(dataset_name="")
    with pytest.raises(ValueError, match="num_clients"):
        ComparisonConfig(dataset_name="adult_income", seed=1, num_clients=1, alpha=1.0)
