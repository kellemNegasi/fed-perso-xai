from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

from fed_perso_xai.cli import main
from fed_perso_xai.data.serialization import load_client_datasets
from fed_perso_xai.explainers import DEFAULT_EXPLAINER_REGISTRY, SHAPExplainer, load_explainer_hyperparameter_grid
from fed_perso_xai.fl.client import ClientData
from fed_perso_xai.models import create_model
from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.orchestration.explanations import (
    LocalExplanationDataset,
    generate_client_local_explanations,
    instantiate_explainer,
)
from fed_perso_xai.utils.config import (
    ArtifactPaths,
    DataPreparationConfig,
    FederatedTrainingConfig,
    LogisticRegressionConfig,
    PartitionConfig,
)
from fed_perso_xai.utils.paths import federated_run_dir


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


class FakeMulticlassKernelExplainer:
    instances: list["FakeMulticlassKernelExplainer"] = []

    def __init__(self, predict_fn, background):
        self.predict_fn = predict_fn
        self.background = np.asarray(background, dtype=np.float64)
        self.expected_value = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)
        self.last_kwargs: dict[str, object] = {}
        type(self).instances.append(self)

    def shap_values(self, X, silent=True, **kwargs):
        self.last_kwargs = dict(kwargs)
        X_arr = np.asarray(X, dtype=np.float64)
        return [
            np.full_like(X_arr, -1.0),
            X_arr + 0.25,
            X_arr + 1.25,
        ]


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


def _install_multiclass_fake_shap(monkeypatch) -> None:
    FakeMulticlassKernelExplainer.instances.clear()
    module = SimpleNamespace(
        KernelExplainer=FakeMulticlassKernelExplainer,
        SamplingExplainer=FakeMulticlassKernelExplainer,
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


class _CountingBinaryProbabilityModel:
    _estimator_type = "classifier"
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self) -> None:
        self.predict_calls = 0
        self.predict_numeric_calls = 0
        self.predict_proba_calls = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.predict_proba_calls += 1
        X_arr = np.asarray(X, dtype=float).reshape(-1, 2)
        positive = np.clip(0.2 + 0.5 * X_arr[:, 0] + 0.1 * X_arr[:, 1], 1.0e-6, 1.0 - 1.0e-6)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.predict_calls += 1
        X_arr = np.asarray(X, dtype=float).reshape(-1, 2)
        return (X_arr[:, 0] + X_arr[:, 1] >= 0.75).astype(int)

    def predict_numeric(self, X: np.ndarray) -> np.ndarray:
        self.predict_numeric_calls += 1
        return self.predict(X)


class _StringMulticlassProbabilityModel:
    _estimator_type = "classifier"
    classes_ = np.asarray(["red", "green", "blue"], dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float).reshape(-1, 2)
        scores = np.column_stack(
            [
                0.3 + 0.1 * X_arr[:, 0],
                0.2 + 0.2 * X_arr[:, 0] + 0.3 * X_arr[:, 1],
                0.4 + 0.1 * X_arr[:, 0] + 1.2 * X_arr[:, 1],
            ]
        )
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class _SignedRegressionModel:
    _estimator_type = "regressor"

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float).reshape(-1, 2)
        return X_arr[:, 0] - 2.0 * X_arr[:, 1]


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
    assert explanation["metadata"]["background_data_source"] == "client_local_train"
    assert explanation["metadata"]["background_sample_size"] == 5
    assert "baseline_instance" not in explanation["metadata"]
    assert "true_label" in explanation["metadata"]
    assert "explained_class" in explanation["metadata"]

    fake_explainer = FakeKernelExplainer.instances[-1]
    assert fake_explainer.last_kwargs["nsamples"] == 32
    assert fake_explainer.last_kwargs["l1_reg"] == "num_features(2)"


def test_shap_avoids_duplicate_numeric_inference_when_probabilities_exist(monkeypatch) -> None:
    _install_fake_shap(monkeypatch)
    dataset = LocalExplanationDataset(
        X_train=np.asarray([[0.0, 0.0], [1.0, 0.5], [0.3, 1.2]], dtype=float),
        y_train=np.asarray([0, 1, 1], dtype=np.int64),
        feature_names=["feature_0", "feature_1"],
    )
    model = _CountingBinaryProbabilityModel()
    explainer = SHAPExplainer(
        config={
            "name": "shap",
            "type": "shap",
            "experiment": {
                "explanation": {
                    "background_data_source": "client_local_train",
                    "random_state": 5,
                    "background_sample_size": 3,
                    "shap_explainer_type": "kernel",
                }
            },
        },
        model=model,
        dataset=dataset,
    )
    explainer.fit(dataset.X_train, dataset.y_train)

    explanation = explainer.explain_instance(np.asarray([0.6, 0.4], dtype=float))

    assert explanation["metadata"]["explained_class"] == 1
    assert model.predict_calls == 1
    assert model.predict_numeric_calls == 0
    assert model.predict_proba_calls == 1


def test_shap_supports_string_multiclass_labels_when_selecting_class_outputs(monkeypatch) -> None:
    _install_multiclass_fake_shap(monkeypatch)
    dataset = LocalExplanationDataset(
        X_train=np.asarray([[0.0, 0.0], [0.2, 0.4], [0.1, 1.0]], dtype=float),
        y_train=np.asarray(["red", "green", "blue"], dtype=object),
        feature_names=["feature_0", "feature_1"],
    )
    model = _StringMulticlassProbabilityModel()
    explainer = SHAPExplainer(
        config={
            "name": "shap",
            "type": "shap",
            "experiment": {
                "explanation": {
                    "background_data_source": "client_local_train",
                    "random_state": 7,
                    "background_sample_size": 3,
                    "shap_explainer_type": "kernel",
                }
            },
        },
        model=model,
        dataset=dataset,
    )
    explainer.fit(dataset.X_train, dataset.y_train)

    instance = np.asarray([0.2, 1.1], dtype=float)
    explanation = explainer.explain_instance(instance)

    assert explanation["prediction"] == "blue"
    assert explanation["metadata"]["explained_class"] == 2
    assert explanation["metadata"]["expected_value"] == 0.6
    np.testing.assert_allclose(np.asarray(explanation["attributions"], dtype=float), instance + 1.25)


def test_shap_permutation_fallback_preserves_attribution_sign() -> None:
    dataset = LocalExplanationDataset(
        X_train=np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=float),
        y_train=np.asarray([0.0, 1.0, -1.0], dtype=float),
        feature_names=["feature_0", "feature_1"],
    )
    explainer = SHAPExplainer(
        config={
            "type": "shap",
            "experiment": {"explanation": {"random_state": 3}},
        },
        model=_SignedRegressionModel(),
        dataset=dataset,
    )

    explanation = explainer.explain_instance(np.asarray([1.0, 1.0], dtype=float))
    attributions = np.asarray(explanation["attributions"], dtype=float)

    assert attributions[0] > 0.0
    assert attributions[1] < 0.0


def test_explain_shap_cli_writes_output(monkeypatch, mock_openml, tmp_path) -> None:
    _install_fake_shap(monkeypatch)
    mock_openml("adult_income")
    paths = ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )
    prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="adult_income",
            seed=13,
            paths=paths,
            partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
        )
    )

    client = load_client_datasets(
        paths.partition_root / "adult_income" / "3_clients" / "alpha_1.0" / "seed_13",
        3,
    )[0]
    model = create_model(
        "logistic_regression",
        n_features=client.train.X.shape[1],
        config=LogisticRegressionConfig(epochs=3, batch_size=4, learning_rate=0.1),
    )
    model.fit(client.train.X, client.train.y, seed=13)

    result_dir = federated_run_dir(paths, "adult_income", 3, 1.0, 13)
    result_dir.mkdir(parents=True, exist_ok=True)
    model.save(result_dir / "model_parameters.npz")
    config = FederatedTrainingConfig(
        dataset_name="adult_income",
        seed=13,
        paths=paths,
        model=LogisticRegressionConfig(epochs=3, batch_size=4, learning_rate=0.1),
        num_clients=3,
        alpha=1.0,
        rounds=1,
        simulation_backend="debug-sequential",
    )
    (result_dir / "config_snapshot.json").write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )

    output_path = tmp_path / "cli_explanations.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fed-perso-xai",
            "explain-shap",
            "--dataset",
            "adult_income",
            "--num-clients",
            "3",
            "--alpha",
            "1.0",
            "--seed",
            "13",
            "--client-id",
            "0",
            "--partition-root",
            str(paths.partition_root),
            "--prepared-root",
            str(paths.prepared_root),
            "--federated-root",
            str(paths.federated_root),
            "--centralized-root",
            str(paths.centralized_root),
            "--comparison-root",
            str(paths.comparison_root),
            "--cache-dir",
            str(paths.cache_dir),
            "--output",
            str(output_path),
            "--max-instances",
            "2",
            "--background-sample-size",
            "3",
            "--random-state",
            "5",
        ],
    )
    main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["method"] == "shap"
    assert payload["client_id"] == 0
    assert payload["split_name"] == "test"
    assert payload["n_explanations"] == min(2, client.test.X.shape[0])


def test_explain_shap_cli_uses_recorded_stage_b_partition_root(
    monkeypatch,
    mock_openml,
    tmp_path,
) -> None:
    _install_fake_shap(monkeypatch)
    mock_openml("adult_income")
    paths = ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )
    alternate_partition_root = tmp_path / "datasets_alt"
    prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="adult_income",
            seed=13,
            paths=ArtifactPaths(
                prepared_root=paths.prepared_root,
                partition_root=alternate_partition_root,
                centralized_root=paths.centralized_root,
                federated_root=paths.federated_root,
                comparison_root=paths.comparison_root,
                cache_dir=paths.cache_dir,
            ),
            partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
        )
    )

    persisted_partition_root = alternate_partition_root / "adult_income" / "3_clients" / "alpha_1.0" / "seed_13"
    client = load_client_datasets(persisted_partition_root, 3)[0]
    model = create_model(
        "logistic_regression",
        n_features=client.train.X.shape[1],
        config=LogisticRegressionConfig(epochs=3, batch_size=4, learning_rate=0.1),
    )
    model.fit(client.train.X, client.train.y, seed=13)

    result_dir = federated_run_dir(paths, "adult_income", 3, 1.0, 13)
    training_dir = result_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    model.save(result_dir / "model_parameters.npz")
    config = FederatedTrainingConfig(
        dataset_name="adult_income",
        seed=13,
        paths=paths,
        model=LogisticRegressionConfig(epochs=3, batch_size=4, learning_rate=0.1),
        num_clients=3,
        alpha=1.0,
        rounds=1,
        simulation_backend="debug-sequential",
    )
    (result_dir / "config_snapshot.json").write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )
    (training_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "partition_data_root": str(persisted_partition_root.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "cli_explanations_recorded_partition.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fed-perso-xai",
            "explain-shap",
            "--dataset",
            "adult_income",
            "--num-clients",
            "3",
            "--alpha",
            "1.0",
            "--seed",
            "13",
            "--client-id",
            "0",
            "--partition-root",
            str(paths.partition_root),
            "--prepared-root",
            str(paths.prepared_root),
            "--federated-root",
            str(paths.federated_root),
            "--centralized-root",
            str(paths.centralized_root),
            "--comparison-root",
            str(paths.comparison_root),
            "--cache-dir",
            str(paths.cache_dir),
            "--output",
            str(output_path),
            "--max-instances",
            "2",
            "--background-sample-size",
            "3",
            "--random-state",
            "5",
        ],
    )
    main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["method"] == "shap"
    assert payload["client_id"] == 0
    assert payload["split_name"] == "test"
    assert payload["row_ids"] == client.test.row_ids[:2].tolist()
