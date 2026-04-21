from __future__ import annotations

import warnings

import numpy as np
import pytest

from fed_perso_xai.explainers import (
    DEFAULT_EXPLAINER_REGISTRY,
    CausalSHAPExplainer,
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


def test_causal_shap_yaml_config_and_grid_are_exposed() -> None:
    spec = DEFAULT_EXPLAINER_REGISTRY.get("causal_shap")
    explanation_cfg = spec["params"]["experiment"]["explanation"]
    assert spec["type"] == "causal_shap"
    assert explanation_cfg["background_data_source"] == "client_local_train"
    assert explanation_cfg["background_sample_size"] == 100
    assert explanation_cfg["causal_shap_corr_threshold"] == 0.3
    assert explanation_cfg["causal_shap_coalitions"] == 20
    assert explanation_cfg["causal_shap_min_graph_samples_warning"] == 10

    grid = load_explainer_hyperparameter_grid()
    assert "causal_shap" in grid
    assert grid["causal_shap"]["background_sample_size"] == [50, 100, 150]
    assert grid["causal_shap"]["causal_shap_coalitions"] == [20, 30, 40]
    assert grid["causal_shap"]["causal_shap_corr_threshold"] == [0.1, 0.3, 0.5]


def test_causal_shap_explainer_instantiation_and_local_background_sampling(
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
        "causal_shap",
        model=model,
        dataset=dataset,
        params_override={
            "background_sample_size": 6,
            "random_state": 7,
            "causal_shap_coalitions": 8,
            "causal_shap_corr_threshold": 0.25,
        },
    )

    assert isinstance(explainer, CausalSHAPExplainer)
    assert explainer.get_info()["background_sample_size"] == 6
    assert explainer.get_info()["background_data_source"] == "client_local_train"
    assert explainer._background is not None
    assert explainer._background.shape == (6, client_data.X_train.shape[1])
    for row in explainer._background:
        assert any(np.array_equal(row, candidate) for candidate in client_data.X_train)

    with pytest.warns(RuntimeWarning, match="small client-local background sample"):
        explanation = explainer.explain_instance(client_data.X_test[0])
    assert explanation["metadata"]["baseline_source"] == "background_mean"
    np.testing.assert_allclose(
        np.asarray(explanation["metadata"]["baseline_instance"], dtype=float),
        np.mean(explainer._background, axis=0),
    )
    assert np.isfinite(float(explanation["metadata"]["baseline_prediction"]))
    assert explanation["metadata"]["graph_inference_sample_size"] == 6
    assert explanation["metadata"]["graph_inference_small_sample_warning"] is True
    assert explanation["metadata"]["graph_inference_small_sample_threshold"] == 10
    assert explanation["metadata"]["graph_inference_fallback"] is None
    assert "graph_inference_warning_messages" in explanation["metadata"]

    explainer_again = instantiate_explainer(
        "causal_shap",
        model=model,
        dataset=dataset,
        params_override={
            "background_sample_size": 6,
            "random_state": 7,
            "causal_shap_coalitions": 8,
            "causal_shap_corr_threshold": 0.25,
        },
    )
    np.testing.assert_allclose(explainer._background, explainer_again._background)


def test_generate_client_local_causal_shap_explanations_schema_and_metadata(
    synthetic_client_splits,
) -> None:
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    feature_names = [f"feature_{idx}" for idx in range(client_data.X_train.shape[1])]

    with pytest.warns(RuntimeWarning, match="small client-local background sample"):
        payload = generate_client_local_explanations(
            client_data=client_data,
            model=model,
            feature_names=feature_names,
            explainer_name="causal_shap",
            split_name="test",
            params_override={
                "background_sample_size": 5,
                "random_state": 9,
                "sampling_strategy": "sequential",
                "max_instances": 3,
                "causal_shap_coalitions": 8,
                "causal_shap_corr_threshold": 0.25,
            },
        )

    assert payload["method"] == "causal_shap"
    assert payload["client_id"] == client_data.client_id
    assert payload["split_name"] == "test"
    assert payload["n_explanations"] == 3
    assert payload["row_ids"] == client_data.row_ids_test[:3].tolist()
    assert payload["info"]["background_sample_size"] == 5
    assert payload["info"]["background_data_source"] == "client_local_train"
    assert payload["info"]["client_context"]["background_dataset_size"] == client_data.X_train.shape[0]

    explanation = payload["explanations"][0]
    assert explanation["method"] == "causal_shap"
    assert explanation["feature_names"] == feature_names
    assert len(explanation["instance"]) == client_data.X_test.shape[1]
    assert len(explanation["attributions"]) == client_data.X_test.shape[1]
    assert len(explanation["prediction_proba"]) == 2
    assert explanation["metadata"]["coalition_samples"] == 8
    assert explanation["metadata"]["correlation_threshold"] == 0.25
    assert explanation["metadata"]["baseline_source"] == "background_mean"
    assert len(explanation["metadata"]["baseline_instance"]) == client_data.X_test.shape[1]
    assert "baseline_prediction" in explanation["metadata"]
    assert explanation["metadata"]["background_data_source"] == "client_local_train"
    assert explanation["metadata"]["background_sample_size"] == 5
    assert explanation["metadata"]["graph_inference_sample_size"] == 5
    assert explanation["metadata"]["graph_inference_small_sample_warning"] is True
    assert explanation["metadata"]["graph_inference_small_sample_threshold"] == 10
    assert explanation["metadata"]["graph_inference_fallback"] is None
    assert "graph_inference_warning_messages" in explanation["metadata"]
    assert set(explanation["metadata"]["causal_graph"]) == set(feature_names)
    assert "explained_class" in explanation["metadata"]
    assert "true_label" in explanation["metadata"]


def test_causal_shap_small_background_warns_and_records_metadata(
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
        "causal_shap",
        model=model,
        dataset=dataset,
        params_override={
            "background_sample_size": 4,
            "random_state": 5,
            "causal_shap_coalitions": 6,
            "causal_shap_min_graph_samples_warning": 8,
        },
    )

    with pytest.warns(RuntimeWarning, match="small client-local background sample"):
        explanation = explainer.explain_instance(client_data.X_test[0])

    metadata = explanation["metadata"]
    assert metadata["graph_inference_sample_size"] == 4
    assert metadata["graph_inference_small_sample_warning"] is True
    assert metadata["graph_inference_small_sample_threshold"] == 8
    assert metadata["graph_inference_fallback"] is None
    assert metadata["graph_inference_nan_count"] == 0
    assert metadata["graph_inference_warning_messages"]


def test_causal_shap_normal_background_does_not_warn(
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
        "causal_shap",
        model=model,
        dataset=dataset,
        params_override={
            "background_sample_size": 8,
            "random_state": 3,
            "causal_shap_coalitions": 6,
            "causal_shap_min_graph_samples_warning": 4,
        },
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        explanation = explainer.explain_instance(client_data.X_test[0])

    assert not any(
        "small client-local background sample" in str(item.message) for item in captured
    )
    metadata = explanation["metadata"]
    assert metadata["graph_inference_sample_size"] == 8
    assert metadata["graph_inference_small_sample_warning"] is False
    assert metadata["graph_inference_small_sample_threshold"] == 4
    assert metadata["graph_inference_fallback"] is None
    assert metadata["graph_inference_warning_messages"] == []


def test_causal_shap_tiny_background_falls_back_to_empty_graph(
    synthetic_client_splits,
) -> None:
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    tiny_background = np.asarray(client_data.X_train[:1], dtype=np.float64)
    dataset = LocalExplanationDataset(
        X_train=tiny_background,
        y_train=np.asarray(client_data.y_train[:1], dtype=np.int64),
        feature_names=[f"feature_{idx}" for idx in range(tiny_background.shape[1])],
    )
    explainer = instantiate_explainer(
        "causal_shap",
        model=model,
        dataset=dataset,
        params_override={
            "background_sample_size": 1,
            "random_state": 2,
            "causal_shap_coalitions": 4,
            "causal_shap_min_graph_samples_warning": 3,
        },
    )

    with pytest.warns(RuntimeWarning) as captured:
        explanation = explainer.explain_instance(client_data.X_test[0])

    metadata = explanation["metadata"]
    assert any(
        "falling back to an empty causal graph" in str(item.message) for item in captured
    )
    assert metadata["graph_inference_sample_size"] == 1
    assert metadata["graph_inference_small_sample_warning"] is True
    assert metadata["graph_inference_fallback"] == "empty_graph_insufficient_rows"
    assert metadata["causal_graph"] == {name: [] for name in dataset.feature_names}
    assert metadata["graph_inference_warning_messages"]


def test_causal_shap_rejects_empty_background_sample_request(
    synthetic_client_splits,
) -> None:
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    dataset = LocalExplanationDataset(
        X_train=client_data.X_train,
        y_train=client_data.y_train,
        feature_names=[f"feature_{idx}" for idx in range(client_data.X_train.shape[1])],
    )

    with pytest.raises(ValueError, match="background_sample_size must be greater than 0"):
        instantiate_explainer(
            "causal_shap",
            model=model,
            dataset=dataset,
            params_override={"background_sample_size": 0},
        )
