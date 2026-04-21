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


class LinearProbModel:
    def __init__(self, weights, bias=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        logits = X_arr @ self.weights + self.bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


class SoftmaxModel:
    def __init__(self, weights, bias):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = np.asarray(bias, dtype=float)

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        logits = X_arr @ self.weights.T + self.bias
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits)
        return exps / np.sum(exps, axis=1, keepdims=True)


class QuadraticInteractionModel:
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


class DatasetStub:
    def __init__(self, X_train):
        self.X_train = np.asarray(X_train, dtype=float)


def _explanation(
    instance,
    attributions,
    *,
    baseline=None,
    prediction=None,
    prediction_proba=None,
    metadata=None,
):
    instance_arr = np.asarray(instance, dtype=float)
    payload = {
        "instance": instance_arr.tolist(),
        "attributions": np.asarray(attributions, dtype=float).tolist(),
        "metadata": dict(metadata or {}),
    }
    if baseline is not None:
        payload["metadata"]["baseline_instance"] = np.asarray(baseline, dtype=float).tolist()
    if prediction is not None:
        payload["prediction"] = prediction
    if prediction_proba is not None:
        payload["prediction_proba"] = np.asarray(prediction_proba, dtype=float).tolist()
    return payload


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


def _manual_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ranks_a = _rankdata(a)
    ranks_b = _rankdata(b)
    diff_a = ranks_a - np.mean(ranks_a)
    diff_b = ranks_b - np.mean(ranks_b)
    denom = np.sqrt(np.sum(diff_a**2) * np.sum(diff_b**2))
    if denom == 0.0:
        return None
    return float(np.sum(diff_a * diff_b) / denom)


def _manual_model_prediction(model, instance) -> float:
    batch = np.asarray(instance, dtype=float).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(batch)).ravel()
        if proba.size == 2:
            return float(proba[1])
        return float(proba.max())
    return float(np.asarray(model.predict(batch)).ravel()[0])


def _manual_original_monotonicity(
    model,
    instance,
    attributions,
    *,
    baseline,
    nr_samples=1,
    features_in_step=1,
    abs_attributions=True,
    normalise=True,
    noise_scale=0.0,
    random_state=None,
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

    if attrs.size != instance_arr.size:
        return None

    original_prediction = _manual_model_prediction(model, instance_arr)
    inv_pred = 1.0 if abs(original_prediction) < eps else (1.0 / abs(original_prediction)) ** 2
    feature_scale = np.maximum(np.abs(instance_arr), 1e-3)
    sorted_indices = np.argsort(attrs)
    rng = np.random.default_rng(random_state)

    att_sums = []
    variance_terms = []
    baseline_arr = np.asarray(baseline, dtype=float).reshape(-1)
    for step in range(int(np.ceil(attrs.size / features_in_step))):
        start = step * features_in_step
        stop = min((step + 1) * features_in_step, attrs.size)
        step_indices = sorted_indices[start:stop]
        if step_indices.size == 0:
            continue

        preds = []
        for _ in range(nr_samples):
            perturbed = instance_arr.copy()
            replacement = baseline_arr[step_indices].copy()
            if noise_scale > 0.0:
                noise = rng.normal(loc=0.0, scale=noise_scale, size=step_indices.size)
                replacement = replacement + noise * feature_scale[step_indices]
            perturbed[step_indices] = replacement
            preds.append(_manual_model_prediction(model, perturbed))

        preds_arr = np.asarray(preds, dtype=float)
        preds_arr = preds_arr[np.isfinite(preds_arr)]
        if preds_arr.size == 0:
            continue

        att_sums.append(float(np.sum(attrs[step_indices])))
        variance_terms.append(float(np.mean((preds_arr - original_prediction) ** 2)) * inv_pred)

    if len(att_sums) < 2 or len(variance_terms) < 2:
        return None

    att_vec = np.asarray(att_sums, dtype=float)
    var_vec = np.asarray(variance_terms, dtype=float)
    if not np.all(np.isfinite(att_vec)) or not np.all(np.isfinite(var_vec)):
        return None
    return _manual_spearman(att_vec, var_vec)


def _manual_cumulative_variant(
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
    groups = [
        ranking[start : start + features_in_step]
        for start in range(0, ranking.size, features_in_step)
    ]
    original_prediction = _manual_model_prediction(model, instance_arr)
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
        new_prediction = _manual_model_prediction(model, perturbed)
        cumulative_mass.append(running_mass)
        cumulative_variance.append(((new_prediction - original_prediction) ** 2) * inv_pred)

    return _manual_spearman(np.asarray(cumulative_mass), np.asarray(cumulative_variance))


def _single_feature_effects(model, instance):
    instance_arr = np.asarray(instance, dtype=float)
    original = _manual_model_prediction(model, instance_arr)
    baseline = np.zeros_like(instance_arr)
    effects = []
    for idx in range(instance_arr.size):
        perturbed = instance_arr.copy()
        perturbed[idx] = baseline[idx]
        new_prediction = _manual_model_prediction(model, perturbed)
        effects.append((original - new_prediction) ** 2)
    return np.asarray(effects, dtype=float)


def test_monotonicity_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("monotonicity_correlation")

    assert isinstance(metric, MonotonicityEvaluator)
    assert metric.nr_samples == 5
    assert metric.features_in_step == 1
    assert metric.abs_attributions is True
    assert metric.normalise is True


def test_monotonicity_matches_original_reference_for_probability_models() -> None:
    model = LinearProbModel(weights=[0.9, 0.5, 0.1])
    instance = np.array([1.0, 0.3, -0.4], dtype=float)
    baseline = np.zeros_like(instance)
    effects = _single_feature_effects(model, instance)
    explanation_results = {
        "method": "shap",
        "explanations": [
            _explanation(
                instance,
                effects,
                baseline=baseline,
                prediction_proba=[0.99, 0.01],
                metadata={"explained_class": 0},
            )
        ],
    }

    evaluator = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        default_baseline=0.0,
        noise_scale=0.0,
        random_state=0,
    )
    score = evaluator.evaluate(model, explanation_results)["monotonicity"]
    expected = _manual_original_monotonicity(
        model,
        instance,
        effects,
        baseline=baseline,
        nr_samples=1,
        features_in_step=1,
        abs_attributions=True,
        normalise=True,
        noise_scale=0.0,
        random_state=0,
    )

    assert expected == pytest.approx(1.0, abs=1e-6)
    assert score == pytest.approx(expected, abs=1e-12)


def test_monotonicity_uses_original_raw_ordering_when_abs_is_disabled() -> None:
    model = LinearModel(weights=[3.0, 1.0, -2.0])
    instance = np.array([2.0, -1.0, 0.5], dtype=float)
    baseline = np.zeros_like(instance)
    attributions = np.array([1.0, -3.0, 2.0], dtype=float)

    explanation_results = {
        "method": "shap",
        "explanations": [_explanation(instance, attributions, baseline=baseline)],
    }

    evaluator = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        default_baseline=0.0,
        abs_attributions=False,
        normalise=False,
        noise_scale=0.0,
        random_state=0,
    )
    score = evaluator.evaluate(model, explanation_results)["monotonicity"]
    expected = _manual_original_monotonicity(
        model,
        instance,
        attributions,
        baseline=baseline,
        nr_samples=1,
        features_in_step=1,
        abs_attributions=False,
        normalise=False,
        noise_scale=0.0,
        random_state=0,
    )
    cumulative = _manual_cumulative_variant(
        model,
        instance,
        attributions,
        baseline=baseline,
        features_in_step=1,
        abs_attributions=False,
        normalise=False,
    )

    assert expected == pytest.approx(0.0, abs=1e-12)
    assert score == pytest.approx(expected, abs=1e-12)
    assert cumulative == pytest.approx(1.0, abs=1e-12)


def test_monotonicity_restores_per_step_groups_instead_of_cumulative_subsets() -> None:
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
    score = evaluator.evaluate(
        model,
        {
            "method": "shap",
            "explanations": [_explanation(instance, scrambled_attributions, baseline=baseline)],
        },
    )["monotonicity"]
    expected = _manual_original_monotonicity(
        model,
        instance,
        scrambled_attributions,
        baseline=baseline,
        nr_samples=1,
        features_in_step=1,
        abs_attributions=True,
        normalise=False,
        noise_scale=0.0,
        random_state=0,
    )
    cumulative = _manual_cumulative_variant(
        model,
        instance,
        scrambled_attributions,
        baseline=baseline,
        features_in_step=1,
        abs_attributions=True,
        normalise=False,
    )

    assert score == pytest.approx(expected, abs=1e-12)
    assert expected == pytest.approx(-0.5, abs=1e-12)
    assert cumulative == pytest.approx(0.5, abs=1e-12)


def test_monotonicity_ignores_dataset_mean_baselines_and_uses_original_default() -> None:
    model = LinearModel(weights=[1.0, -2.0, 0.5])
    instance = np.array([2.0, 1.0, -3.0], dtype=float)
    attributions = np.array([0.4, 0.7, 0.1], dtype=float)
    dataset = DatasetStub(X_train=np.full((5, 3), 10.0))

    evaluator = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        default_baseline=0.0,
        abs_attributions=True,
        normalise=True,
        noise_scale=0.0,
        random_state=0,
    )
    score = evaluator.evaluate(
        model,
        {"method": "shap", "explanations": [_explanation(instance, attributions)]},
        dataset=dataset,
    )["monotonicity"]
    expected = _manual_original_monotonicity(
        model,
        instance,
        attributions,
        baseline=np.zeros_like(instance),
        nr_samples=1,
        features_in_step=1,
        abs_attributions=True,
        normalise=True,
        noise_scale=0.0,
        random_state=0,
    )
    dataset_mean_variant = _manual_original_monotonicity(
        model,
        instance,
        attributions,
        baseline=np.mean(dataset.X_train, axis=0),
        nr_samples=1,
        features_in_step=1,
        abs_attributions=True,
        normalise=True,
        noise_scale=0.0,
        random_state=0,
    )

    assert score == pytest.approx(expected, abs=1e-12)
    assert dataset_mean_variant == pytest.approx(1.0, abs=1e-12)
    assert expected == pytest.approx(0.8660254037844387, abs=1e-12)


def test_monotonicity_handles_edge_cases_from_original_semantics() -> None:
    linear_model = LinearModel(weights=[0.7, -0.3, 0.2])
    single_feature_model = LinearModel(weights=[0.7])
    constant_model = ConstantModel(value=1.5)

    zero_attr_score = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        random_state=3,
    ).evaluate(
        linear_model,
        {
            "method": "integrated_gradients",
            "explanations": [
                _explanation([1.0, -2.0, 3.0], [0.0, 0.0, 0.0], baseline=[0.0, 0.0, 0.0])
            ],
        },
    )["monotonicity"]
    constant_pred_score = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        random_state=5,
    ).evaluate(
        constant_model,
        {
            "method": "lime",
            "explanations": [
                _explanation([1.0, 2.0, 3.0], [0.9, 0.2, 0.1], baseline=[0.0, 0.0, 0.0])
            ],
        },
    )["monotonicity"]
    single_feature_score = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        random_state=7,
    ).evaluate(
        single_feature_model,
        {
            "method": "causal_shap",
            "explanations": [_explanation([2.0], [0.8], baseline=[0.0])],
        },
    )["monotonicity"]
    oversized_group_score = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=8,
        random_state=9,
    ).evaluate(
        linear_model,
        {
            "method": "shap",
            "explanations": [
                _explanation([1.0, 2.0, 3.0], [0.5, 0.3, 0.1], baseline=[0.0, 0.0, 0.0])
            ],
        },
    )["monotonicity"]
    nan_attr_score = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        random_state=11,
    ).evaluate(
        linear_model,
        {
            "method": "shap",
            "explanations": [
                _explanation([1.0, 2.0, 3.0], [0.5, np.nan, 0.1], baseline=[0.0, 0.0, 0.0])
            ],
        },
    )["monotonicity"]

    assert zero_attr_score == pytest.approx(0.0, abs=1e-12)
    assert constant_pred_score == pytest.approx(0.0, abs=1e-12)
    assert single_feature_score == pytest.approx(0.0, abs=1e-12)
    assert oversized_group_score == pytest.approx(0.0, abs=1e-12)
    assert nan_attr_score == pytest.approx(0.0, abs=1e-12)


def test_monotonicity_handles_tied_attributions_with_original_argsort_behavior() -> None:
    model = LinearModel(weights=[0.4, 0.6, -0.8, 0.2])
    instance = np.array([1.0, -2.0, 0.5, 1.5], dtype=float)
    baseline = np.zeros_like(instance)
    attributions = np.array([0.2, 0.2, 0.9, 0.9], dtype=float)

    evaluator = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=2,
        default_baseline=0.0,
        abs_attributions=True,
        normalise=False,
        noise_scale=0.0,
        random_state=0,
    )
    score = evaluator.evaluate(
        model,
        {"method": "shap", "explanations": [_explanation(instance, attributions, baseline=baseline)]},
    )["monotonicity"]
    expected = _manual_original_monotonicity(
        model,
        instance,
        attributions,
        baseline=baseline,
        nr_samples=1,
        features_in_step=2,
        abs_attributions=True,
        normalise=False,
        noise_scale=0.0,
        random_state=0,
    )

    assert expected is not None
    assert np.isfinite(score)
    assert score == pytest.approx(expected, abs=1e-12)


def test_monotonicity_uses_original_multiclass_max_probability_target() -> None:
    model = SoftmaxModel(
        weights=[
            [3.0, 0.5, -1.0],
            [0.1, 2.0, 0.2],
            [-0.5, -1.0, 1.5],
        ],
        bias=[0.3, -0.2, 0.1],
    )
    instance = np.array([1.2, -0.7, 0.4], dtype=float)
    baseline = np.zeros_like(instance)
    effects = _single_feature_effects(model, instance)

    evaluator = MonotonicityEvaluator(
        nr_samples=1,
        features_in_step=1,
        default_baseline=0.0,
        abs_attributions=True,
        normalise=True,
        noise_scale=0.0,
        random_state=0,
    )
    score = evaluator.evaluate(
        model,
        {
            "method": "shap",
            "explanations": [
                _explanation(
                    instance,
                    effects,
                    baseline=baseline,
                    prediction_proba=[0.1, 0.8, 0.1],
                    metadata={"explained_class": 1},
                )
            ],
        },
    )["monotonicity"]
    expected = _manual_original_monotonicity(
        model,
        instance,
        effects,
        baseline=baseline,
        nr_samples=1,
        features_in_step=1,
        abs_attributions=True,
        normalise=True,
        noise_scale=0.0,
        random_state=0,
    )

    assert expected == pytest.approx(1.0, abs=1e-12)
    assert score == pytest.approx(expected, abs=1e-12)
