from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fed_perso_xai.fl.client import (
    apply_shared_parameter_payload,
    extract_shared_parameter_payload,
)
from fed_perso_xai.fl.strategy import _build_secure_aggregator, _weighted_average_parameter_sets
from fed_perso_xai.models import load_global_model
from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.orchestration.training import train_federated_from_prepared
from fed_perso_xai.utils.config import (
    ArtifactPaths,
    DataPreparationConfig,
    FederatedTrainingConfig,
    LogisticRegressionConfig,
    PartitionConfig,
)

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


def test_shared_parameter_helpers_keep_stage1_payloads_explicit() -> None:
    parameters = [
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([3.0], dtype=np.float64),
    ]

    shared_payload = extract_shared_parameter_payload(parameters)
    merged = apply_shared_parameter_payload(
        current_parameters=[
            np.array([-9.0, -9.0], dtype=np.float64),
            np.array([-9.0], dtype=np.float64),
        ],
        shared_parameters=shared_payload.shared_parameters,
        shared_parameter_indices=shared_payload.shared_parameter_indices,
    )

    assert shared_payload.shared_parameter_indices == (0, 1)
    np.testing.assert_allclose(merged[0], parameters[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(merged[1], parameters[1], atol=0.0, rtol=0.0)


def test_secure_aggregator_matches_plain_shared_weighted_average() -> None:
    config = FederatedTrainingConfig(
        dataset_name="adult_income",
        secure_aggregation=True,
        secure_num_helpers=5,
        secure_privacy_threshold=2,
        secure_reconstruction_threshold=3,
        secure_quantization_scale=100_000,
        secure_seed=17,
    )
    secure_aggregator = _build_secure_aggregator(config)
    parameter_sets = [
        [np.array([0.25, 1.0]), np.array([0.5])],
        [np.array([1.25, -0.5]), np.array([1.0])],
        [np.array([-0.75, 0.25]), np.array([-0.5])],
    ]
    weights = [4, 3, 5]

    plain_average = _weighted_average_parameter_sets(parameter_sets, weights)
    weighted_payloads = [
        [array * weight for array in payload]
        for payload, weight in zip(parameter_sets, weights, strict=True)
    ]
    secure_result = secure_aggregator.aggregate(weighted_payloads, round_id=9)
    secure_average = [tensor / sum(weights) for tensor in secure_result.aggregated_tensors]

    np.testing.assert_allclose(secure_average[0], plain_average[0], atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(secure_average[1], plain_average[1], atol=1e-5, rtol=0.0)


@pytest.mark.skipif(
    not FLOWER_AVAILABLE,
    reason="Flower is not installed; secure federated training integration needs Flower.",
)
def test_debug_federated_training_supports_plain_and_secure_modes(
    mock_openml,
    tmp_path,
) -> None:
    mock_openml("adult_income")
    paths = _build_paths(tmp_path)
    prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="adult_income",
            seed=31,
            paths=paths,
            partition=PartitionConfig(
                num_clients=3,
                alpha=1.0,
                min_client_samples=2,
                max_retries=20,
            ),
        )
    )

    plain_artifacts, plain_summary = train_federated_from_prepared(
        FederatedTrainingConfig(
            dataset_name="adult_income",
            seed=31,
            paths=paths,
            model=LogisticRegressionConfig(epochs=2, batch_size=4, learning_rate=0.1),
            num_clients=3,
            alpha=1.0,
            rounds=2,
            simulation_backend="debug-sequential",
            secure_aggregation=False,
        )
    )
    secure_artifacts, secure_summary = train_federated_from_prepared(
        FederatedTrainingConfig(
            dataset_name="adult_income",
            seed=31,
            paths=paths,
            model=LogisticRegressionConfig(epochs=2, batch_size=4, learning_rate=0.1),
            num_clients=3,
            alpha=1.0,
            rounds=2,
            simulation_backend="debug-sequential",
            secure_aggregation=True,
            secure_num_helpers=5,
            secure_privacy_threshold=2,
            secure_reconstruction_threshold=3,
            secure_quantization_scale=100_000,
            secure_seed=31,
        )
    )

    assert plain_summary["round_history_summary"][0]["aggregation"]["mode"] == "plain"
    assert secure_summary["round_history_summary"][0]["aggregation"]["mode"] == "secure"
    assert secure_summary["round_history_summary"][0]["aggregation"]["helper_count"] == 5

    plain_parameters = load_global_model(plain_artifacts.run_dir).model.get_parameters()
    secure_parameters = load_global_model(secure_artifacts.run_dir).model.get_parameters()
    np.testing.assert_allclose(
        np.asarray(secure_parameters[0], dtype=np.float64),
        np.asarray(plain_parameters[0], dtype=np.float64),
        atol=1e-4,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(secure_parameters[1], dtype=np.float64),
        np.asarray(plain_parameters[1], dtype=np.float64),
        atol=1e-4,
        rtol=0.0,
    )
