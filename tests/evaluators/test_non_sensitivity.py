from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.evaluators import make_metric
from fed_perso_xai.evaluators.non_sensitivity import NonSensitivityEvaluator


class AffineBinaryProbModel:
    def __init__(self, positive_weights, intercept=0.0):
        self.positive_weights = np.asarray(positive_weights, dtype=float)
        self.intercept = float(intercept)

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        positive = np.clip(self.intercept + X_arr @ self.positive_weights, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


class ThreeClassFeatureModel:
    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        scores = np.column_stack(
            [
                1.0 + 2.0 * X_arr[:, 0],
                4.0 + 0.1 * X_arr[:, 1],
                np.ones(X_arr.shape[0], dtype=float),
            ]
        )
        return scores / np.sum(scores, axis=1, keepdims=True)


class ConstantModel:
    def __init__(self, value=0.4):
        self.value = float(value)

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.full(X_arr.shape[0], self.value, dtype=float)


def _explanation(instance, attributions, *, model=None, baseline=None, explained_class=None):
    instance_arr = np.asarray(instance, dtype=float)
    explanation = {
        "instance": instance_arr.tolist(),
        "attributions": np.asarray(attributions, dtype=float).tolist(),
        "metadata": {},
    }
    if baseline is not None:
        explanation["metadata"]["baseline_instance"] = np.asarray(baseline, dtype=float).tolist()
    if explained_class is not None:
        explanation["metadata"]["explained_class"] = int(explained_class)

    if model is not None:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(instance_arr.reshape(1, -1))[0]
            explanation["prediction_proba"] = proba.tolist()
            explanation["prediction"] = int(np.argmax(proba))
        elif hasattr(model, "predict"):
            explanation["prediction"] = float(model.predict(instance_arr.reshape(1, -1))[0])
    return explanation


def test_non_sensitivity_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("non_sensitivity")

    assert isinstance(metric, NonSensitivityEvaluator)
    assert metric.zero_threshold == pytest.approx(1.0e-05)
    assert metric.delta_tolerance == pytest.approx(1.0e-04)
    assert metric.features_per_step == 1


def test_non_sensitivity_selects_low_attribution_features_and_counts_violations() -> None:
    model = AffineBinaryProbModel(positive_weights=[0.0, 0.4, 0.0, 0.0], intercept=0.2)
    instance = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
    explanation = _explanation(
        instance,
        [0.0, 1.0e-6, 0.5, 0.2],
        model=model,
        baseline=[0.0, 0.0, 0.0, 0.0],
    )
    explanation_results = {"method": "shap", "explanations": [explanation]}

    evaluator = NonSensitivityEvaluator(
        zero_threshold=1.0e-5,
        delta_tolerance=0.1,
        features_per_step=1,
        default_baseline=0.0,
    )
    scores = evaluator.evaluate(model, explanation_results)

    assert scores["non_sensitivity_violation_fraction"] == pytest.approx(0.5, abs=1e-12)
    assert scores["non_sensitivity_safe_fraction"] == pytest.approx(0.5, abs=1e-12)
    assert scores["non_sensitivity_zero_features"] == pytest.approx(2.0, abs=1e-12)
    assert scores["non_sensitivity_delta_mean"] == pytest.approx(0.2, abs=1e-12)


def test_non_sensitivity_uses_target_class_probability_and_grouping() -> None:
    model = ThreeClassFeatureModel()
    instance = np.array([1.0, 0.0], dtype=float)
    explanation = _explanation(
        instance,
        [0.0, 0.4],
        model=model,
        baseline=[0.0, 0.0],
        explained_class=0,
    )
    explanation_results = {"method": "causal_shap", "explanations": [explanation]}

    evaluator = NonSensitivityEvaluator(
        zero_threshold=1.0e-5,
        delta_tolerance=0.1,
        features_per_step=5,
    )
    scores = evaluator.evaluate(model, explanation_results)

    original = model.predict_proba(instance.reshape(1, -1))[0, 0]
    perturbed = model.predict_proba(np.array([[0.0, 0.0]], dtype=float))[0, 0]
    expected_delta = abs(original - perturbed)

    assert scores["non_sensitivity_violation_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert scores["non_sensitivity_safe_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert scores["non_sensitivity_zero_features"] == pytest.approx(1.0, abs=1e-12)
    assert scores["non_sensitivity_delta_mean"] == pytest.approx(expected_delta, abs=1e-12)


def test_non_sensitivity_handles_requested_edge_cases_deterministically() -> None:
    no_zero_model = AffineBinaryProbModel(positive_weights=[0.2, -0.1], intercept=0.5)
    no_zero_results = {
        "method": "lime",
        "explanations": [
            _explanation(
                [1.0, 2.0],
                [0.3, -0.2],
                model=no_zero_model,
                baseline=[0.0, 0.0],
            )
        ],
    }

    constant_model = ConstantModel(value=0.7)
    all_zero_results = {
        "method": "integrated_gradients",
        "explanations": [
            _explanation(
                [2.0, -1.0],
                [0.0, 0.0],
                model=constant_model,
                baseline=[0.0, 0.0],
            )
        ],
    }

    single_feature_model = AffineBinaryProbModel(positive_weights=[0.6], intercept=0.1)
    single_feature_results = {
        "method": "shap",
        "explanations": [
            _explanation(
                [1.0],
                [0.0],
                model=single_feature_model,
                baseline=[0.0],
            )
        ],
    }

    evaluator = NonSensitivityEvaluator(
        zero_threshold=1.0e-5,
        delta_tolerance=0.05,
        features_per_step=10,
    )

    no_zero_scores = evaluator.evaluate(no_zero_model, no_zero_results)
    constant_scores_a = evaluator.evaluate(constant_model, all_zero_results)
    constant_scores_b = evaluator.evaluate(constant_model, all_zero_results)
    single_feature_scores = evaluator.evaluate(single_feature_model, single_feature_results)

    assert no_zero_scores == {
        "non_sensitivity_violation_fraction": 0.0,
        "non_sensitivity_safe_fraction": 0.0,
        "non_sensitivity_zero_features": 0.0,
        "non_sensitivity_delta_mean": 0.0,
    }

    assert constant_scores_a["non_sensitivity_violation_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert constant_scores_a["non_sensitivity_safe_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert constant_scores_a["non_sensitivity_zero_features"] == pytest.approx(2.0, abs=1e-12)
    assert constant_scores_a["non_sensitivity_delta_mean"] == pytest.approx(0.0, abs=1e-12)
    assert constant_scores_a == constant_scores_b

    single_feature_original = single_feature_model.predict_proba(np.array([[1.0]], dtype=float))[0, 1]
    single_feature_perturbed = single_feature_model.predict_proba(np.array([[0.0]], dtype=float))[0, 1]
    expected_single_delta = abs(single_feature_original - single_feature_perturbed)

    assert single_feature_scores["non_sensitivity_violation_fraction"] == pytest.approx(1.0, abs=1e-12)
    assert single_feature_scores["non_sensitivity_safe_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert single_feature_scores["non_sensitivity_zero_features"] == pytest.approx(1.0, abs=1e-12)
    assert single_feature_scores["non_sensitivity_delta_mean"] == pytest.approx(
        expected_single_delta,
        abs=1e-12,
    )
