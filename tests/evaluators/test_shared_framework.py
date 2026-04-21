from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fed_perso_xai.evaluators import (
    DEFAULT_METRIC_REGISTRY,
    MetricCapabilities,
    MetricInput,
    baseline_vector,
    build_metric_rng,
    chunk_indices,
    evaluate_metrics_for_method,
    extract_attribution_vector,
    extract_instance_vector,
    extract_prediction_value,
    generate_random_masked_batch,
    load_metric_config,
    make_metric,
    mask_feature_indices,
    metric_capabilities,
    model_prediction,
    model_predictions,
    prediction_label,
    prepare_attributions,
    sample_random_mask_indices,
    support_indices,
    top_k_mask_indices,
    InfidelityEvaluator,
)


class PredictProbaModel:
    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        positive = np.clip(0.2 + 0.3 * X_arr[:, 0] - 0.1 * X_arr[:, 1], 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


class DecisionFunctionModel:
    def decision_function(self, X):
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, 0] - 2.0 * X_arr[:, 1]


class PredictOnlyModel:
    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr[:, 0] > X_arr[:, 1]).astype(float)


class DummyInstanceMetric(MetricCapabilities):
    metric_names = ("instance_metric",)

    def _evaluate(self, metric_input: MetricInput):
        cache = metric_input.cache_bucket("dummy")
        cache["calls"] = int(cache.get("calls", 0)) + 1
        explanation = metric_input.current_explanation()
        assert explanation is not None
        return {
            "instance_metric": float(metric_input.explanation_idx),
            "cache_calls": float(cache["calls"]),
        }


class DummyBatchMetric(MetricCapabilities):
    per_instance = False
    requires_full_batch = True
    metric_names = ("batch_metric",)

    def _evaluate(self, metric_input: MetricInput):
        return {"batch_metric": float(len(metric_input.explanations))}


def _explanation_payload():
    explanations = [
        {
            "method": "shap",
            "instance": [1.0, 2.0, 3.0],
            "attributions": [0.3, -0.2, 0.1],
            "feature_names": ["f0", "f1", "f2"],
            "prediction": 1,
            "prediction_proba": [0.3, 0.7],
            "metadata": {
                "target": 1,
                "baseline_instance": [0.0, 0.0, 0.0],
            },
        },
        {
            "method": "shap",
            "instance": [0.5, 1.5, 2.5],
            "attributions": [0.1, 0.4, -0.2],
            "feature_names": ["f0", "f1", "f2"],
            "prediction": 0,
            "prediction_proba": [0.8, 0.2],
            "metadata": {
                "target": 0,
                "baseline_instance": [0.0, 0.0, 0.0],
            },
        },
    ]
    return {
        "method": "shap",
        "client_id": 3,
        "split_name": "test",
        "row_ids": ["row-0", "row-1"],
        "explanations": explanations,
    }


def test_metric_input_construction_and_index_helpers() -> None:
    payload = _explanation_payload()
    metric_input = MetricInput.from_results(
        model=PredictProbaModel(),
        explanation_results=payload,
        dataset=SimpleNamespace(feature_names=["d0", "d1", "d2"]),
    )

    assert metric_input.method == "shap"
    assert metric_input.current_explanation() is None
    assert tuple(metric_input.feature_names()) == ("d0", "d1", "d2")

    indexed = metric_input.with_index(1)
    assert indexed.current_explanation() == payload["explanations"][1]
    assert tuple(indexed.feature_names()) == ("f0", "f1", "f2")
    assert list(indexed.iter_explanations())[0][0] == 0


def test_attribution_and_prediction_helpers_follow_explanation_schema() -> None:
    explanation = _explanation_payload()["explanations"][0]

    attrs = extract_attribution_vector(explanation)
    instance = extract_instance_vector(explanation)
    assert attrs is not None
    assert instance is not None
    np.testing.assert_allclose(attrs, np.array([0.3, -0.2, 0.1]))
    np.testing.assert_allclose(instance, np.array([1.0, 2.0, 3.0]))

    prepared = prepare_attributions(attrs, abs_attributions=True, normalise=True)
    np.testing.assert_allclose(prepared, np.array([1.0, 2.0 / 3.0, 1.0 / 3.0]))

    assert extract_prediction_value(explanation) == pytest.approx(0.7)
    assert prediction_label(explanation) == 1


def test_prediction_helpers_choose_probabilities_then_decision_function_then_predict() -> None:
    batch = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    proba_values = model_predictions(PredictProbaModel(), batch)
    assert proba_values.shape == (2,)
    assert proba_values[0] > proba_values[1]
    assert model_prediction(PredictProbaModel(), batch[0]) == pytest.approx(proba_values[0])

    decision_values = model_predictions(DecisionFunctionModel(), batch, prefer_probability=True)
    np.testing.assert_allclose(decision_values, np.array([1.0, -2.0]))

    predict_values = model_predictions(PredictOnlyModel(), batch, prefer_probability=True)
    np.testing.assert_allclose(predict_values, np.array([1.0, 0.0]))


def test_baseline_and_perturbation_helpers_are_reproducible() -> None:
    payload = _explanation_payload()
    explanation = payload["explanations"][0]
    instance = extract_instance_vector(explanation)
    assert instance is not None

    dataset = SimpleNamespace(
        X_train=np.array(
            [
                [1.0, 3.0, 5.0],
                [3.0, 5.0, 7.0],
            ],
            dtype=float,
        )
    )

    baseline = baseline_vector(explanation, instance, default_baseline=-1.0, dataset=dataset)
    np.testing.assert_allclose(baseline, np.zeros_like(instance))

    no_metadata = {
        "instance": instance.tolist(),
        "attributions": explanation["attributions"],
        "metadata": {},
    }
    dataset_baseline = baseline_vector(no_metadata, instance, default_baseline=-1.0, dataset=dataset)
    np.testing.assert_allclose(dataset_baseline, np.array([2.0, 4.0, 6.0]))

    masked = mask_feature_indices(instance, np.array([0, 2]), np.zeros_like(instance))
    np.testing.assert_allclose(masked, np.array([0.0, 2.0, 0.0]))

    support = support_indices(np.array([0.4, 0.01, 0.2, 0.0]), magnitude_threshold=0.05, min_features=2)
    np.testing.assert_array_equal(support, np.array([0, 2]))

    top_k = top_k_mask_indices(np.array([0.4, -0.8, 0.2]), 2)
    np.testing.assert_array_equal(top_k, np.array([1, 0]))

    rng_a = build_metric_rng(7)
    rng_b = build_metric_rng(7)
    sampled_a = sample_random_mask_indices(rng_a, n_features=5, mask_size=2)
    sampled_b = sample_random_mask_indices(rng_b, n_features=5, mask_size=2)
    np.testing.assert_array_equal(sampled_a, sampled_b)

    batch_a = generate_random_masked_batch(
        instance,
        np.zeros_like(instance),
        n_trials=3,
        mask_size=2,
        rng=build_metric_rng(11),
    )
    batch_b = generate_random_masked_batch(
        instance,
        np.zeros_like(instance),
        n_trials=3,
        mask_size=2,
        rng=build_metric_rng(11),
    )
    np.testing.assert_allclose(batch_a, batch_b)

    groups = chunk_indices(np.array([0, 1, 2, 3]), features_per_step=3)
    assert [group.tolist() for group in groups] == [[0, 1, 2], [3]]


def test_execution_plumbing_routes_per_instance_and_batch_metrics() -> None:
    payload = _explanation_payload()
    result = evaluate_metrics_for_method(
        metric_objs={
            "instance_metric": DummyInstanceMetric(),
            "batch_metric": DummyBatchMetric(),
        },
        metric_caps={
            "instance_metric": metric_capabilities(DummyInstanceMetric()),
            "batch_metric": metric_capabilities(DummyBatchMetric()),
        },
        explainer=None,
        expl_results=payload,
        dataset_mapping={
            100: (0, payload["explanations"][0]),
            101: (1, payload["explanations"][1]),
        },
        model=PredictProbaModel(),
        dataset=None,
        method_label="shap",
        log_progress=False,
    )

    assert result.batch_metrics == {"batch_metric": 2.0}
    assert result.instance_metrics[100][0]["instance_metric"] == 0.0
    assert result.instance_metrics[101][1]["instance_metric"] == 1.0
    assert result.instance_metrics[101][1]["cache_calls"] == 2.0


def test_metric_registry_and_config_support_are_present() -> None:
    spec = DEFAULT_METRIC_REGISTRY.get("infidelity")
    assert spec["module"] == "fed_perso_xai.evaluators.infidelity"
    assert spec["class"] == "InfidelityEvaluator"

    loaded = load_metric_config()
    assert "correctness" in loaded
    assert loaded["contrastivity_ssim"]["params"]["similarity_func"] == (
        "fed_perso_xai.evaluators.utils.structural_similarity"
    )
    assert isinstance(make_metric("infidelity"), InfidelityEvaluator)
