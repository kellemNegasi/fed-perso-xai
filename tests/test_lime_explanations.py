from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.explainers import (
    DEFAULT_EXPLAINER_REGISTRY,
    LIMEExplainer,
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


def test_lime_yaml_config_and_grid_are_exposed() -> None:
    spec = DEFAULT_EXPLAINER_REGISTRY.get("lime")
    explanation_cfg = spec["params"]["experiment"]["explanation"]
    assert spec["type"] == "lime"
    assert explanation_cfg["background_data_source"] == "client_local_train"
    assert explanation_cfg["lime_num_samples"] == 100
    assert explanation_cfg["lime_noise_scale"] == 0.1
    assert explanation_cfg["lime_kernel_width"] == 2.0
    assert float(explanation_cfg["lime_alpha"]) == 1e-2
    assert explanation_cfg["lime_target_class"] is None

    grid = load_explainer_hyperparameter_grid()
    assert "lime" in grid
    assert grid["lime"]["lime_num_samples"] == [50, 100, 200]
    assert grid["lime"]["lime_kernel_width"] == [1.5, 2.0, 3.0]


def test_lime_explainer_instantiation_uses_client_local_training_stats(
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
        "lime",
        model=model,
        dataset=dataset,
        params_override={
            "random_state": 7,
            "lime_num_samples": 24,
            "lime_kernel_width": 1.5,
            "lime_noise_scale": 0.05,
            "lime_alpha": 0.02,
        },
    )

    assert isinstance(explainer, LIMEExplainer)
    assert explainer.get_info()["background_data_source"] == "client_local_train"
    assert explainer.get_info()["baseline_source"] == "train_mean"
    assert explainer._train_mean is not None
    assert explainer._train_std is not None
    np.testing.assert_allclose(explainer._train_mean, np.mean(client_data.X_train, axis=0))


def test_lime_rejects_non_local_background_source(synthetic_client_splits) -> None:
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    dataset = LocalExplanationDataset(
        X_train=client_data.X_train,
        y_train=client_data.y_train,
        feature_names=[f"feature_{idx}" for idx in range(client_data.X_train.shape[1])],
    )

    with pytest.raises(ValueError, match="client-local explainer background data"):
        instantiate_explainer(
            "lime",
            model=model,
            dataset=dataset,
            params_override={"background_data_source": "global_shared_train"},
        )


def test_generate_client_local_lime_explanations_schema_and_metadata(
    synthetic_client_splits,
) -> None:
    client_data = _build_client_data(synthetic_client_splits)
    model = _fit_model(client_data)
    feature_names = [f"feature_{idx}" for idx in range(client_data.X_train.shape[1])]

    payload = generate_client_local_explanations(
        client_data=client_data,
        model=model,
        feature_names=feature_names,
        explainer_name="lime",
        split_name="test",
        params_override={
            "random_state": 9,
            "sampling_strategy": "sequential",
            "max_instances": 3,
            "lime_num_samples": 32,
            "lime_kernel_width": 1.5,
            "lime_noise_scale": 0.05,
            "lime_alpha": 0.02,
        },
    )

    assert payload["method"] == "lime"
    assert payload["client_id"] == client_data.client_id
    assert payload["split_name"] == "test"
    assert payload["n_explanations"] == 3
    assert payload["row_ids"] == client_data.row_ids_test[:3].tolist()
    assert payload["info"]["background_data_source"] == "client_local_train"
    assert payload["info"]["baseline_source"] == "train_mean"
    assert payload["info"]["client_context"]["background_dataset_size"] == client_data.X_train.shape[0]

    explanation = payload["explanations"][0]
    assert explanation["method"] == "lime"
    assert explanation["feature_names"] == feature_names
    assert len(explanation["instance"]) == client_data.X_test.shape[1]
    assert len(explanation["attributions"]) == client_data.X_test.shape[1]
    assert len(explanation["prediction_proba"]) == 2
    assert explanation["metadata"]["background_data_source"] == "client_local_train"
    assert explanation["metadata"]["baseline_source"] == "train_mean"
    assert explanation["metadata"]["num_samples"] == 32
    assert explanation["metadata"]["kernel_width"] == 1.5
    assert explanation["metadata"]["noise_scale"] == 0.05
    assert explanation["metadata"]["alpha"] == 0.02
    assert "baseline_prediction" in explanation["metadata"]
    assert "explained_class" in explanation["metadata"]
    assert "true_label" in explanation["metadata"]
    np.testing.assert_allclose(
        np.asarray(explanation["metadata"]["baseline_instance"], dtype=float),
        np.mean(client_data.X_train, axis=0),
    )
    assert any(abs(value) > 0.0 for value in explanation["attributions"])


class _BinarySwitchingModel:
    _estimator_type = "classifier"
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float).reshape(-1, 2)
        positive = np.clip(0.15 + 0.6 * X_arr[:, 0], 1.0e-6, 1.0 - 1.0e-6)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_numeric(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)


class _MulticlassSwitchingModel:
    _estimator_type = "classifier"
    classes_ = np.asarray([0, 1, 2], dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float).reshape(-1, 2)
        scores = np.column_stack(
            [
                2.0 - X_arr[:, 0],
                1.5 + X_arr[:, 0] - X_arr[:, 1],
                1.0 + 1.2 * X_arr[:, 1],
            ]
        )
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_numeric(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)


def _toy_lime_dataset() -> LocalExplanationDataset:
    X_train = np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.3],
            [1.2, 0.2],
            [0.2, 1.5],
        ],
        dtype=float,
    )
    y_train = np.asarray([0, 0, 1, 2], dtype=np.int64)
    return LocalExplanationDataset(
        X_train=X_train,
        y_train=y_train,
        feature_names=["feature_0", "feature_1"],
    )


def _make_lime_explainer(model, **expl_cfg) -> LIMEExplainer:
    config = {
        "name": "lime",
        "type": "lime",
        "experiment": {
            "explanation": {
                "random_state": 3,
                "lime_num_samples": 8,
                "lime_noise_scale": 0.05,
                "lime_kernel_width": 1.5,
                "lime_alpha": 0.01,
                **expl_cfg,
            }
        },
    }
    explainer = LIMEExplainer(config=config, model=model, dataset=_toy_lime_dataset())
    explainer.fit(explainer.dataset.X_train, explainer.dataset.y_train)
    return explainer


def test_lime_binary_uses_fixed_positive_class_convention_consistently() -> None:
    explainer = _make_lime_explainer(_BinarySwitchingModel())
    instance = np.asarray([0.2, 0.0], dtype=float)

    explanation = explainer.explain_instance(instance)
    perturbations = np.asarray(
        [
            instance,
            [1.4, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    perturbation_targets = explainer._local_target_vector(
        perturbations,
        explainer.model.predict(perturbations),
        target_class=explanation["metadata"]["explained_class"],
    )

    assert int(np.argmax(np.asarray(explanation["prediction_proba"], dtype=float))) == 0
    assert explanation["metadata"]["explained_class"] == 1
    np.testing.assert_allclose(
        perturbation_targets,
        explainer.model.predict_proba(perturbations)[:, 1],
    )


def test_lime_multiclass_keeps_original_instance_target_class_fixed() -> None:
    explainer = _make_lime_explainer(_MulticlassSwitchingModel())
    instance = np.asarray([0.1, 1.2], dtype=float)

    explanation = explainer.explain_instance(instance)
    perturbations = np.asarray(
        [
            instance,
            [2.4, -0.2],
            [-0.5, 0.1],
        ],
        dtype=float,
    )
    proba = explainer.model.predict_proba(perturbations)
    perturbation_targets = explainer._local_target_vector(
        perturbations,
        explainer.model.predict(perturbations),
        target_class=explanation["metadata"]["explained_class"],
    )
    dynamic_targets = proba[np.arange(len(proba)), np.argmax(proba, axis=1)]

    assert explanation["metadata"]["explained_class"] == int(
        np.argmax(np.asarray(explanation["prediction_proba"], dtype=float))
    )
    np.testing.assert_allclose(
        perturbation_targets,
        proba[:, explanation["metadata"]["explained_class"]],
    )
    assert not np.allclose(perturbation_targets, dynamic_targets)


def test_lime_explicit_target_class_overrides_original_prediction() -> None:
    explainer = _make_lime_explainer(_MulticlassSwitchingModel(), lime_target_class=1)
    instance = np.asarray([0.1, 1.2], dtype=float)

    explanation = explainer.explain_instance(instance)
    perturbations = np.asarray(
        [
            instance,
            [2.4, -0.2],
            [-0.5, 0.1],
        ],
        dtype=float,
    )
    perturbation_targets = explainer._local_target_vector(
        perturbations,
        explainer.model.predict(perturbations),
        target_class=explanation["metadata"]["explained_class"],
    )

    assert explanation["metadata"]["explained_class"] == 1
    np.testing.assert_allclose(
        perturbation_targets,
        explainer.model.predict_proba(perturbations)[:, 1],
    )
