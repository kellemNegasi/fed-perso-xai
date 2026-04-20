from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from fed_perso_xai.explainers import DEFAULT_EXPLAINER_REGISTRY, SHAPExplainer, load_explainer_hyperparameter_grid
from fed_perso_xai.fl.client import ClientData
from fed_perso_xai.models import create_model
from fed_perso_xai.orchestration.explanations import (
    LocalExplanationDataset,
    generate_client_local_explanations,
    instantiate_explainer,
)
from fed_perso_xai.utils.config import LogisticRegressionConfig


class FakeKernelExplainer:
    instances: list["FakeKernelExplainer"] = []

    def __init__(self, predict_fn, background):
        self.predict_fn = predict_fn
        self.background = np.asarray(background, dtype=np.float64)
        self.expected_value = np.asarray([0.25, 0.75], dtype=np.float64)
        self.last_kwargs: dict[str, object] = {}
        type(self).instances.append(self)

    def shap_values(self, X, silent=True, **kwargs):
        self.last_kwargs = dict(kwargs)
        X_arr = np.asarray(X, dtype=np.float64)
        return [np.zeros_like(X_arr), X_arr + 0.5]


class FakeSamplingExplainer(FakeKernelExplainer):
    instances: list["FakeSamplingExplainer"] = []


class FakeTreeExplainer:
    instances: list["FakeTreeExplainer"] = []

    def __init__(self, model):
        self.model = model
        self.expected_value = np.asarray([0.1, 0.9], dtype=np.float64)
        self.last_kwargs: dict[str, object] = {}
        type(self).instances.append(self)

    def shap_values(self, X, silent=True, **kwargs):
        self.last_kwargs = dict(kwargs)
        X_arr = np.asarray(X, dtype=np.float64)
        return [np.zeros_like(X_arr), X_arr]


def _install_fake_shap(monkeypatch) -> None:
    FakeKernelExplainer.instances.clear()
    FakeSamplingExplainer.instances.clear()
    FakeTreeExplainer.instances.clear()
    module = SimpleNamespace(
        KernelExplainer=FakeKernelExplainer,
        SamplingExplainer=FakeSamplingExplainer,
        TreeExplainer=FakeTreeExplainer,
    )
    monkeypatch.setitem(sys.modules, "shap", module)


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


def test_shap_yaml_config_and_grid_are_exposed() -> None:
    spec = DEFAULT_EXPLAINER_REGISTRY.get("shap")
    explanation_cfg = spec["params"]["experiment"]["explanation"]
    assert spec["type"] == "shap"
    assert explanation_cfg["background_data_source"] == "client_local_train"
    assert explanation_cfg["background_sample_size"] == 100
    assert explanation_cfg["shap_explainer_type"] == "kernel"

    grid = load_explainer_hyperparameter_grid()
    assert "shap" in grid
    assert grid["shap"]["background_sample_size"] == [50, 100, 150]
    assert set(grid["shap"]["shap_explainer_type"]) == {"kernel", "sampling"}


def test_shap_explainer_instantiation_and_background_sampling(monkeypatch, synthetic_client_splits) -> None:
    _install_fake_shap(monkeypatch)
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    dataset = LocalExplanationDataset(
        X_train=client_data.X_train,
        y_train=client_data.y_train,
        feature_names=[f"feature_{idx}" for idx in range(client_data.X_train.shape[1])],
    )

    explainer = instantiate_explainer(
        "shap",
        model=model,
        dataset=dataset,
        params_override={
            "background_sample_size": 4,
            "random_state": 7,
            "shap_explainer_type": "kernel",
        },
    )

    assert isinstance(explainer, SHAPExplainer)
    assert explainer.get_info()["background_sample_size"] == 4
    assert explainer.get_info()["background_data_source"] == "client_local_train"
    assert explainer._background is not None
    assert explainer._background.shape == (4, client_data.X_train.shape[1])
    assert len(FakeKernelExplainer.instances) == 1
    np.testing.assert_allclose(FakeKernelExplainer.instances[0].background, explainer._background)

    explainer_again = instantiate_explainer(
        "shap",
        model=model,
        dataset=dataset,
        params_override={
            "background_sample_size": 4,
            "random_state": 7,
            "shap_explainer_type": "kernel",
        },
    )
    np.testing.assert_allclose(explainer._background, explainer_again._background)


def test_generate_client_local_shap_explanations_schema_and_metadata(monkeypatch, synthetic_client_splits) -> None:
    _install_fake_shap(monkeypatch)
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    feature_names = [f"feature_{idx}" for idx in range(client_data.X_train.shape[1])]

    payload = generate_client_local_explanations(
        client_data=client_data,
        model=model,
        feature_names=feature_names,
        explainer_name="shap",
        split_name="test",
        params_override={
            "background_sample_size": 5,
            "random_state": 9,
            "sampling_strategy": "sequential",
            "max_instances": 3,
            "shap_explainer_type": "kernel",
            "shap_nsamples": 32,
            "shap_l1_reg": "num_features",
            "shap_l1_reg_k": 2,
        },
    )

    assert payload["method"] == "shap"
    assert payload["client_id"] == client_data.client_id
    assert payload["split_name"] == "test"
    assert payload["n_explanations"] == 3
    assert payload["row_ids"] == client_data.row_ids_test[:3].tolist()
    assert payload["info"]["background_sample_size"] == 5
    assert payload["info"]["background_data_source"] == "client_local_train"
    assert payload["info"]["client_context"]["background_dataset_size"] == client_data.X_train.shape[0]

    explanation = payload["explanations"][0]
    assert explanation["method"] == "shap"
    assert explanation["feature_names"] == feature_names
    assert len(explanation["instance"]) == client_data.X_test.shape[1]
    assert len(explanation["attributions"]) == client_data.X_test.shape[1]
    assert len(explanation["prediction_proba"]) == 2
    assert "expected_value" in explanation["metadata"]
    assert "target" in explanation["metadata"]

    fake_explainer = FakeKernelExplainer.instances[-1]
    assert fake_explainer.last_kwargs["nsamples"] == 32
    assert fake_explainer.last_kwargs["l1_reg"] == "num_features(2)"
