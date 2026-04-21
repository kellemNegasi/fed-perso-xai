from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fed_perso_xai.evaluators import build_metric_rng, make_metric, sample_random_mask_indices
from fed_perso_xai.evaluators.infidelity import InfidelityEvaluator


class LinearModel:
    def __init__(self, weights, bias=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.weights + self.bias


class AffineBinaryProbModel:
    def __init__(self, positive_weights, intercept=0.4):
        self.positive_weights = np.asarray(positive_weights, dtype=float)
        self.intercept = float(intercept)

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        positive = self.intercept + X_arr @ self.positive_weights
        return np.column_stack([1.0 - positive, positive])


class ConstantModel:
    def __init__(self, value=3.0):
        self.value = float(value)

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.full(X_arr.shape[0], self.value, dtype=float)


def _explanation(instance, attributions, *, model=None, baseline=None, target=None):
    instance_arr = np.asarray(instance, dtype=float)
    metadata = {}
    if baseline is not None:
        metadata["baseline_instance"] = np.asarray(baseline, dtype=float).tolist()
    if target is not None:
        metadata["target"] = target

    explanation = {
        "instance": instance_arr.tolist(),
        "attributions": np.asarray(attributions, dtype=float).tolist(),
        "metadata": metadata,
    }

    if model is not None:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(instance_arr.reshape(1, -1))[0]
            explanation["prediction_proba"] = proba.tolist()
            explanation["prediction"] = int(np.argmax(proba))
        elif hasattr(model, "predict"):
            explanation["prediction"] = float(model.predict(instance_arr.reshape(1, -1))[0])
    return explanation


def test_infidelity_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("infidelity")

    assert isinstance(metric, InfidelityEvaluator)
    assert metric.n_perturb_samples == 32
    assert metric.features_per_sample == 3
    assert metric.normalise is True


def test_infidelity_zero_for_linear_model_with_exact_attributions() -> None:
    model = LinearModel(weights=[0.6, -0.2, 0.3])
    instance = np.array([1.0, -2.0, 0.5], dtype=float)
    explanation = _explanation(instance, model.weights, model=model, baseline=[0.0, 0.0, 0.0])
    explanation_results = {"method": "shap", "explanations": [explanation]}

    evaluator = InfidelityEvaluator(
        n_perturb_samples=64,
        features_per_sample=2,
        default_baseline=0.0,
        random_state=0,
    )
    score = evaluator.evaluate(model, explanation_results)["infidelity"]

    assert score == pytest.approx(0.0, abs=1e-9)


def test_infidelity_matches_manual_squared_error_aggregation() -> None:
    model = LinearModel(weights=[0.6, -0.2, 0.3])
    instance = np.array([1.0, -2.0, 0.5], dtype=float)
    wrong_attributions = model.weights * np.array([1.0, -3.0, 2.0], dtype=float)
    explanation = _explanation(instance, wrong_attributions, model=model, baseline=[0.0, 0.0, 0.0])
    explanation_results = {"method": "shap", "explanations": [explanation]}

    evaluator = InfidelityEvaluator(
        n_perturb_samples=32,
        features_per_sample=2,
        default_baseline=0.0,
        random_state=1,
    )
    score = evaluator.evaluate(model, explanation_results)["infidelity"]

    rng = build_metric_rng(1, offset=0)
    errors = []
    original = float(model.predict(instance.reshape(1, -1))[0])
    for _ in range(32):
        chosen = sample_random_mask_indices(rng, n_features=instance.size, mask_size=2)
        perturbed = instance.copy()
        perturbed[chosen] = 0.0
        delta = instance - perturbed
        approx_change = float(np.dot(wrong_attributions, delta))
        true_change = float(original - model.predict(perturbed.reshape(1, -1))[0])
        errors.append((approx_change - true_change) ** 2)
    expected = float(np.mean(errors))

    assert score == pytest.approx(expected, abs=1e-12)
    assert score > 0.01


def test_infidelity_uses_target_class_probability_when_available() -> None:
    model = AffineBinaryProbModel(positive_weights=[0.1, 0.05], intercept=0.4)
    instance = np.array([1.0, 2.0], dtype=float)
    class_zero_attributions = np.array([-0.1, -0.05], dtype=float)
    explanation = _explanation(
        instance,
        class_zero_attributions,
        model=model,
        baseline=[0.0, 0.0],
        target=0,
    )
    explanation_results = {"method": "causal_shap", "explanations": [explanation]}

    evaluator = InfidelityEvaluator(
        n_perturb_samples=24,
        features_per_sample=1,
        random_state=4,
    )
    scores = evaluator.evaluate(model, explanation_results)

    assert scores["infidelity"] == pytest.approx(0.0, abs=1e-12)


def test_infidelity_uses_dataset_mean_baseline_when_metadata_is_missing() -> None:
    model = LinearModel(weights=[0.5, -0.25])
    instance = np.array([3.0, 1.0], dtype=float)
    dataset = SimpleNamespace(
        X_train=np.array(
            [
                [1.0, 5.0],
                [3.0, 1.0],
            ],
            dtype=float,
        )
    )
    explanation = {
        "instance": instance.tolist(),
        "attributions": model.weights.tolist(),
        "prediction": float(model.predict(instance.reshape(1, -1))[0]),
        "metadata": {},
    }
    explanation_results = {"method": "lime", "explanations": [explanation]}

    evaluator = InfidelityEvaluator(
        n_perturb_samples=20,
        features_per_sample=1,
        default_baseline=-1.0,
        random_state=9,
    )
    score = evaluator.evaluate(model, explanation_results, dataset=dataset)["infidelity"]

    assert score == pytest.approx(0.0, abs=1e-12)


def test_infidelity_handles_edge_cases_and_is_reproducible() -> None:
    constant_model = ConstantModel(value=2.5)
    zero_attr_explanation = _explanation(
        [1.0, -2.0],
        [0.0, 0.0],
        model=constant_model,
        baseline=[0.0, 0.0],
    )
    zero_attr_results = {"method": "integrated_gradients", "explanations": [zero_attr_explanation]}

    zero_delta_model = LinearModel(weights=[0.7, -0.3])
    zero_delta_instance = np.array([2.0, -1.0], dtype=float)
    zero_delta_explanation = _explanation(
        zero_delta_instance,
        [10.0, -4.0],
        model=zero_delta_model,
        baseline=zero_delta_instance,
    )
    zero_delta_results = {"method": "shap", "explanations": [zero_delta_explanation]}

    tiny_model = LinearModel(weights=[1.0e-6])
    tiny_instance = np.array([2.0e-6], dtype=float)
    tiny_explanation = _explanation(
        tiny_instance,
        [1.0e-6],
        model=tiny_model,
        baseline=[0.0],
    )
    tiny_results = {"method": "shap", "explanations": [tiny_explanation]}

    zero_attr_score = InfidelityEvaluator(
        n_perturb_samples=8,
        features_per_sample=1,
        random_state=3,
    ).evaluate(constant_model, zero_attr_results)["infidelity"]
    zero_delta_score = InfidelityEvaluator(
        n_perturb_samples=8,
        features_per_sample=2,
        random_state=5,
    ).evaluate(zero_delta_model, zero_delta_results)["infidelity"]

    evaluator = InfidelityEvaluator(
        n_perturb_samples=8,
        features_per_sample=5,
        random_state=7,
    )
    tiny_score_a = evaluator.evaluate(tiny_model, tiny_results)["infidelity"]
    tiny_score_b = evaluator.evaluate(tiny_model, tiny_results)["infidelity"]

    assert zero_attr_score == pytest.approx(0.0, abs=1e-12)
    assert zero_delta_score == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(tiny_score_a)
    assert tiny_score_a == pytest.approx(0.0, abs=1e-18)
    assert tiny_score_a == pytest.approx(tiny_score_b, abs=0.0)
