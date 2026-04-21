from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.evaluators import MonotonicityEvaluator, make_metric


class LinearModel:
    def __init__(self, weights, bias=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.weights + self.bias


class QuadraticInteractionModel:
    """
    Small nonlinear model used to exercise cumulative-subset ordering semantics.

    The interaction terms make the cumulative prediction-change sequence depend on
    the chosen ranking, which lets the monotonicity score distinguish a faithful
    ordering from a scrambled one even under the cumulative formulation.
    """

    def __init__(self, linear_weights, interaction_weights):
        self.linear_weights = np.asarray(linear_weights, dtype=float)
        self.interaction_weights = np.asarray(interaction_weights, dtype=float)

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        values = X_arr @ self.linear_weights
        values += self.interaction_weights[0] * X_arr[:, 0] * X_arr[:, 1]
        values += self.interaction_weights[1] * X_arr[:, 0] * X_arr[:, 2]
        values += self.interaction_weights[2] * X_arr[:, 1] * X_arr[:, 2]
        return values


class ConstantModel:
    def __init__(self, value=2.0):
        self.value = float(value)

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.full(X_arr.shape[0], self.value, dtype=float)


def _explanation(instance, attributions, *, model, baseline=None, explained_class=None):
    instance_arr = np.asarray(instance, dtype=float)
    metadata = {}
    if baseline is not None:
        metadata["baseline_instance"] = np.asarray(baseline, dtype=float).tolist()
    if explained_class is not None:
        metadata["explained_class"] = explained_class

    return {
        "instance": instance_arr.tolist(),
        "attributions": np.asarray(attributions, dtype=float).tolist(),
        "metadata": metadata,
        "prediction": float(model.predict(instance_arr.reshape(1, -1))[0]),
    }


def _rankdata(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(arr.size, dtype=float)
    _, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    for val_idx, count in enumerate(counts):
        if count <= 1:
            continue
        tie_positions = np.where(inverse == val_idx)[0]
        ranks[tie_positions] = float(np.mean(ranks[tie_positions]))
    return ranks


def _manual_spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a = _rankdata(a)
    ranks_b = _rankdata(b)
    diff_a = ranks_a - np.mean(ranks_a)
    diff_b = ranks_b - np.mean(ranks_b)
    denom = np.sqrt(np.sum(diff_a**2) * np.sum(diff_b**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(diff_a * diff_b) / denom)


def _manual_monotonicity(
    model,
    instance,
    attributions,
    *,
    baseline,
    features_in_step=1,
    abs_attributions=True,
    normalise=True,
    eps=1.0e-5,
):
    instance_arr = np.asarray(instance, dtype=float)
    attrs = np.asarray(attributions, dtype=float).reshape(-1)
    if abs_attributions:
        attrs = np.abs(attrs)
    if normalise:
        denom = np.max(np.abs(attrs))
        if denom > 0.0:
            attrs = attrs / denom

    ranking = np.argsort(-np.abs(attrs), kind="stable")
    groups = [ranking[start : start + features_in_step] for start in range(0, ranking.size, features_in_step)]
    original_prediction = float(model.predict(instance_arr.reshape(1, -1))[0])
    inv_pred = 1.0 if abs(original_prediction) < eps else (1.0 / abs(original_prediction)) ** 2

    cumulative_mass = []
    cumulative_variance = []
    selected = np.asarray([], dtype=int)
    running_mass = 0.0
    baseline_arr = np.asarray(baseline, dtype=float).reshape(-1)
    for group in groups:
        selected = np.concatenate([selected, np.asarray(group, dtype=int)])
        running_mass += float(np.sum(attrs[group]))
        perturbed = instance_arr.copy()
        perturbed[selected] = baseline_arr[selected]
        new_prediction = float(model.predict(perturbed.reshape(1, -1))[0])
        cumulative_mass.append(running_mass)
        cumulative_variance.append(((new_prediction - original_prediction) ** 2) * inv_pred)

    return _manual_spearman(np.asarray(cumulative_mass), np.asarray(cumulative_variance))


def _single_feature_effects(model, instance):
    instance_arr = np.asarray(instance, dtype=float)
    original = float(model.predict(instance_arr.reshape(1, -1))[0])
    baseline = np.zeros_like(instance_arr)
    effects = []
    for idx in range(instance_arr.size):
        perturbed = instance_arr.copy()
        perturbed[idx] = baseline[idx]
        new_prediction = float(model.predict(perturbed.reshape(1, -1))[0])
        effects.append((original - new_prediction) ** 2)
    return np.asarray(effects, dtype=float)


def test_monotonicity_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("monotonicity_correlation")

    assert isinstance(metric, MonotonicityEvaluator)
    assert metric.nr_samples == 5
    assert metric.features_in_step == 1
    assert metric.abs_attributions is True
    assert metric.normalise is True


def test_monotonicity_matches_manual_cumulative_definition() -> None:
    model = LinearModel(weights=[3.0, 1.0, -2.0])
    instance = np.array([2.0, -1.0, 0.5], dtype=float)
    baseline = np.zeros_like(instance)
    attributions = np.array([-4.0, 1.0, 3.0], dtype=float)

    explanation_results = {
        "method": "shap",
        "explanations": [
            _explanation(
                instance,
                attributions,
                model=model,
                baseline=baseline,
            )
        ],
    }

    evaluator = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        default_baseline=0.0,
        abs_attributions=True,
        normalise=True,
        noise_scale=0.0,
        random_state=0,
    )
    score = evaluator.evaluate(model, explanation_results)["monotonicity"]
    expected = _manual_monotonicity(
        model,
        instance,
        attributions,
        baseline=baseline,
        features_in_step=1,
        abs_attributions=True,
        normalise=True,
    )

    assert score == pytest.approx(expected, abs=1e-12)


def test_monotonicity_prefers_faithful_ordering_over_scrambled_ordering() -> None:
    model = QuadraticInteractionModel(
        linear_weights=[2.1731491, 2.83751693, 2.76500795],
        interaction_weights=[2.439333, 1.644284, -1.00112909],
    )
    instance = np.array([-1.67559444, -0.37103531, -1.07106343], dtype=float)
    baseline = np.zeros_like(instance)

    faithful_attributions = _single_feature_effects(model, instance)
    scrambled_attributions = faithful_attributions[[1, 2, 0]]

    evaluator = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        default_baseline=0.0,
        abs_attributions=True,
        normalise=False,
        noise_scale=0.0,
        random_state=0,
    )

    faithful_score = evaluator.evaluate(
        model,
        {
            "method": "shap",
            "explanations": [
                _explanation(
                    instance,
                    faithful_attributions,
                    model=model,
                    baseline=baseline,
                )
            ],
        },
    )["monotonicity"]
    scrambled_score = evaluator.evaluate(
        model,
        {
            "method": "shap",
            "explanations": [
                _explanation(
                    instance,
                    scrambled_attributions,
                    model=model,
                    baseline=baseline,
                )
            ],
        },
    )["monotonicity"]

    assert faithful_score == pytest.approx(1.0, abs=1e-12)
    assert scrambled_score == pytest.approx(0.5, abs=1e-12)
    assert faithful_score > scrambled_score


def test_monotonicity_handles_edge_cases_and_is_deterministic() -> None:
    linear_model = LinearModel(weights=[0.7, -0.3, 0.2])
    single_feature_model = LinearModel(weights=[0.7])
    constant_model = ConstantModel(value=1.5)

    zero_attr_results = {
        "method": "integrated_gradients",
        "explanations": [
            _explanation(
                [1.0, -2.0, 3.0],
                [0.0, 0.0, 0.0],
                model=linear_model,
                baseline=[0.0, 0.0, 0.0],
            )
        ],
    }
    constant_pred_results = {
        "method": "lime",
        "explanations": [
            _explanation(
                [1.0, 2.0, 3.0],
                [0.9, 0.2, 0.1],
                model=constant_model,
                baseline=[0.0, 0.0, 0.0],
            )
        ],
    }
    single_feature_results = {
        "method": "causal_shap",
        "explanations": [
            _explanation(
                [2.0],
                [0.8],
                model=single_feature_model,
                baseline=[0.0],
            )
        ],
    }

    zero_attr_score = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        random_state=3,
    ).evaluate(linear_model, zero_attr_results)["monotonicity"]
    constant_pred_score = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        random_state=5,
    ).evaluate(constant_model, constant_pred_results)["monotonicity"]
    single_feature_score = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        random_state=7,
    ).evaluate(single_feature_model, single_feature_results)["monotonicity"]

    noisy_results = {
        "method": "shap",
        "explanations": [
            _explanation(
                [1.0, -1.0, 0.5],
                [0.6, 0.4, 0.2],
                model=linear_model,
                baseline=[0.0, 0.0, 0.0],
            )
        ],
    }
    evaluator = MonotonicityEvaluator(
        nr_samples=6,
        features_in_step=1,
        default_baseline=0.0,
        noise_scale=0.05,
        random_state=11,
    )
    noisy_score_a = evaluator.evaluate(linear_model, noisy_results)["monotonicity"]
    noisy_score_b = evaluator.evaluate(linear_model, noisy_results)["monotonicity"]

    assert zero_attr_score == pytest.approx(0.0, abs=1e-12)
    assert constant_pred_score == pytest.approx(0.0, abs=1e-12)
    assert single_feature_score == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(noisy_score_a)
    assert noisy_score_a == pytest.approx(noisy_score_b, abs=0.0)
