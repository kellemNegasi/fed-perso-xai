from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.explainers import (
    DEFAULT_EXPLAINER_REGISTRY,
    IntegratedGradientsExplainer,
    load_explainer_hyperparameter_grid,
)
from fed_perso_xai.fl.client import ClientData
from fed_perso_xai.models import create_model
from fed_perso_xai.orchestration.explanations import (
    LocalExplanationDataset,
    generate_client_local_explanations,
    instantiate_explainer,
)
from fed_perso_xai.utils.config import LogisticRegressionConfig


def _build_client_data(synthetic_client_splits) -> ClientData:
    split = synthetic_client_splits[0]
    return ClientData(
        client_id=split["client_id"],
        X_train=split["X_train"],
        y_train=split["y_train"],
        row_ids_train=split["row_ids_train"],
        X_test=split["X_test"],
        y_test=split["y_test"],
        row_ids_test=split["row_ids_test"],
    )


def _fit_model(client_data: ClientData):
    model = create_model(
        "logistic_regression",
        n_features=client_data.X_train.shape[1],
        config=LogisticRegressionConfig(epochs=5, batch_size=4, learning_rate=0.1),
    )
    model.fit(client_data.X_train, client_data.y_train, seed=11)
    return model


def test_integrated_gradients_yaml_config_and_grid_are_exposed() -> None:
    spec = DEFAULT_EXPLAINER_REGISTRY.get("integrated_gradients")
    explanation_cfg = spec["params"]["experiment"]["explanation"]
    assert spec["type"] == "integrated_gradients"
    assert explanation_cfg["background_data_source"] == "client_local_train"
    assert explanation_cfg["ig_steps"] == 40
    assert float(explanation_cfg["ig_epsilon"]) == 1e-6
    assert explanation_cfg["ig_target_class"] == 1
    assert explanation_cfg["ig_allow_nondifferentiable"] is False

    grid = load_explainer_hyperparameter_grid()
    assert "integrated_gradients" in grid
    assert grid["integrated_gradients"]["ig_steps"] == [20, 40, 60]


def test_integrated_gradients_instantiation_uses_client_local_mean_baseline(
    synthetic_client_splits,
) -> None:
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    dataset = LocalExplanationDataset(
        X_train=client_data.X_train,
        y_train=client_data.y_train,
        feature_names=[f"feature_{idx}" for idx in range(client_data.X_train.shape[1])],
    )

    explainer = instantiate_explainer(
        "integrated_gradients",
        model=model,
        dataset=dataset,
        params_override={
            "random_state": 7,
            "ig_steps": 24,
            "ig_epsilon": 1e-5,
        },
    )

    assert isinstance(explainer, IntegratedGradientsExplainer)
    assert explainer.get_info()["background_data_source"] == "client_local_train"
    assert explainer.get_info()["baseline_source"] == "train_mean"
    assert explainer._train_mean is not None
    np.testing.assert_allclose(explainer._train_mean, np.mean(client_data.X_train, axis=0))


def test_integrated_gradients_rejects_non_local_baseline_source(synthetic_client_splits) -> None:
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    dataset = LocalExplanationDataset(
        X_train=client_data.X_train,
        y_train=client_data.y_train,
        feature_names=[f"feature_{idx}" for idx in range(client_data.X_train.shape[1])],
    )

    with pytest.raises(ValueError, match="client-local explainer background data"):
        instantiate_explainer(
            "integrated_gradients",
            model=model,
            dataset=dataset,
            params_override={"background_data_source": "global_shared_train"},
        )


def test_generate_client_local_integrated_gradients_explanations_schema_and_metadata(
    synthetic_client_splits,
) -> None:
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    feature_names = [f"feature_{idx}" for idx in range(client_data.X_train.shape[1])]

    payload = generate_client_local_explanations(
        client_data=client_data,
        model=model,
        feature_names=feature_names,
        explainer_name="integrated_gradients",
        split_name="test",
        params_override={
            "random_state": 9,
            "sampling_strategy": "sequential",
            "max_instances": 3,
            "ig_steps": 24,
            "ig_epsilon": 1e-5,
            "ig_target_class": 1,
        },
    )

    assert payload["method"] == "integrated_gradients"
    assert payload["client_id"] == client_data.client_id
    assert payload["split_name"] == "test"
    assert payload["n_explanations"] == 3
    assert payload["row_ids"] == client_data.row_ids_test[:3].tolist()
    assert payload["info"]["background_data_source"] == "client_local_train"
    assert payload["info"]["baseline_source"] == "train_mean"
    assert payload["info"]["client_context"]["background_dataset_size"] == client_data.X_train.shape[0]

    explanation = payload["explanations"][0]
    assert explanation["method"] == "integrated_gradients"
    assert explanation["feature_names"] == feature_names
    assert len(explanation["instance"]) == client_data.X_test.shape[1]
    assert len(explanation["attributions"]) == client_data.X_test.shape[1]
    assert len(explanation["prediction_proba"]) == 2
    assert explanation["metadata"]["baseline_source"] == "train_mean"
    assert explanation["metadata"]["n_steps"] == 24
    assert explanation["metadata"]["epsilon"] == 1e-5
    assert explanation["metadata"]["explained_class"] == 1
    assert "true_label" in explanation["metadata"]
    assert any(abs(value) > 0.0 for value in explanation["attributions"])


def test_integrated_gradients_persists_inferred_explained_class_without_explicit_target(
    synthetic_client_splits,
) -> None:
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    dataset = LocalExplanationDataset(
        X_train=client_data.X_train,
        y_train=client_data.y_train,
        feature_names=[f"feature_{idx}" for idx in range(client_data.X_train.shape[1])],
    )

    explainer = instantiate_explainer(
        "integrated_gradients",
        model=model,
        dataset=dataset,
        params_override={
            "random_state": 5,
            "ig_steps": 12,
            "ig_epsilon": 1e-5,
            "ig_target_class": None,
        },
    )

    explanation = explainer.explain_instance(client_data.X_test[0])
    expected_class = int(np.argmax(np.asarray(explanation["prediction_proba"], dtype=float)))

    assert explanation["metadata"]["explained_class"] == expected_class
