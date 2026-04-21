from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.evaluators import make_metric
from fed_perso_xai.evaluators.completeness import CompletenessEvaluator
from fed_perso_xai.evaluators.perturbation import generate_random_masked_batch


class LinearProbModel:
    def __init__(self, weights, bias=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)

    def predict_proba(self, X):
        logits = np.asarray(X, dtype=float) @ self.weights + self.bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


class ThreeClassFeatureModel:
    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        scores = np.column_stack(
            [
                1.0 + 2.0 * X_arr[:, 0] + 0.5 * X_arr[:, 1],
                1.0 + 0.2 * X_arr[:, 0] + 1.5 * X_arr[:, 1],
                1.0 + 0.1 * X_arr[:, 2],
            ]
        )
        scores = np.clip(scores, 1.0e-8, None)
        return scores / np.sum(scores, axis=1, keepdims=True)


def test_completeness_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("completeness_deletion")

    assert isinstance(metric, CompletenessEvaluator)
    assert metric.magnitude_threshold == pytest.approx(1.0e-08)
    assert metric.min_features == 1
    assert metric.random_trials == 10
    assert metric.fast_mode is True


def test_completeness_drop_matches_manual_computation() -> None:
    model = LinearProbModel(weights=[1.0, -0.5, 0.25, 0.0])
    instance = np.array([2.0, 1.0, -1.0, 0.5], dtype=float)
    proba = model.predict_proba(instance.reshape(1, -1))[0]

    explanation = {
        "attributions": [0.9, 0.2, 0.02, 0.0],
        "instance": instance.tolist(),
        "metadata": {"baseline_instance": [0.0, 0.0, 0.0, 0.0]},
        "prediction_proba": proba.tolist(),
    }
    explanation_results = {"method": "shap", "explanations": [explanation]}

    evaluator = CompletenessEvaluator(
        magnitude_threshold=0.05,
        min_features=1,
        random_trials=0,
        default_baseline=0.0,
    )
    scores = evaluator.evaluate(model, explanation_results, dataset=None, explainer=None)

    mask_indices = np.array([0, 1], dtype=int)
    orig_pred = proba[1]
    perturbed = instance.copy()
    perturbed[mask_indices] = 0.0
    new_pred = model.predict_proba(perturbed.reshape(1, -1))[0, 1]
    expected_drop = abs(orig_pred - new_pred) / (abs(orig_pred) + 1e-8)

    # Deterministic drop should match direct model computation with masked features.
    assert scores["completeness_drop"] == pytest.approx(expected_drop, abs=1e-9)
    # Random baselines disabled => random drop must be exactly zero.
    assert scores["completeness_random_drop"] == pytest.approx(0.0, abs=1e-12)
    # Score equals target drop minus random baseline (zero here).
    assert scores["completeness_score"] == pytest.approx(expected_drop, abs=1e-9)


def test_completeness_uses_target_class_and_random_baseline_matches_equal_size_masks() -> None:
    model = ThreeClassFeatureModel()
    instance = np.array([3.0, 1.0, 0.5], dtype=float)
    proba = model.predict_proba(instance.reshape(1, -1))[0]
    baseline = np.zeros_like(instance)

    explanation_results = {
        "method": "causal_shap",
        "explanations": [
            {
                "attributions": [0.8, -0.8, 0.1],
                "instance": instance.tolist(),
                "prediction": 0,
                "prediction_proba": proba.tolist(),
                "metadata": {
                    "target": 0,
                    "baseline_instance": baseline.tolist(),
                },
            }
        ],
    }

    evaluator = CompletenessEvaluator(
        magnitude_threshold=0.2,
        min_features=1,
        random_trials=4,
        default_baseline=0.0,
        fast_mode=False,
        random_state=17,
    )
    scores = evaluator.evaluate(model, explanation_results, dataset=None, explainer=None)

    mask_indices = np.array([0, 1], dtype=int)
    selected = instance.copy()
    selected[mask_indices] = 0.0
    selected_drop = abs(proba[0] - model.predict_proba(selected.reshape(1, -1))[0, 0]) / (
        abs(proba[0]) + 1e-8
    )

    random_batch = generate_random_masked_batch(
        instance,
        baseline,
        n_trials=4,
        mask_size=2,
        rng=np.random.default_rng(17),
    )
    random_preds = model.predict_proba(random_batch)[:, 0]
    expected_random = np.mean(np.abs(proba[0] - random_preds) / (abs(proba[0]) + 1e-8))

    assert scores["completeness_drop"] == pytest.approx(selected_drop, abs=1e-9)
    assert scores["completeness_random_drop"] == pytest.approx(expected_random, abs=1e-9)
    assert scores["completeness_score"] == pytest.approx(
        max(0.0, selected_drop - expected_random),
        abs=1e-9,
    )


def test_completeness_clips_negative_advantage_and_handles_edge_cases() -> None:
    model = LinearProbModel(weights=[1.0, 0.2])
    single_feature_model = LinearProbModel(weights=[1.4])
    explanation_results = {
        "method": "lime",
        "explanations": [
            {
                # Stable tie break should pick feature 0 when min_features enforces one feature.
                "attributions": [0.0, 0.0],
                "instance": [2.0, 1.0],
                "prediction": 1,
                "prediction_proba": model.predict_proba(np.array([[2.0, 1.0]]))[0].tolist(),
                "metadata": {"baseline_instance": [-1.0]},
            },
            {
                # Zero denominator should be skipped to 0.0 rather than exploding.
                "attributions": [0.7, 0.1],
                "instance": [0.5, -0.5],
                "prediction_proba": [1.0, 0.0],
                "metadata": {"baseline_instance": [0.0, 0.0], "target": 1},
            },
            {
                # Missing attributions => empty selected feature set / invalid payload.
                "instance": [1.0, 2.0],
                "prediction_proba": [0.3, 0.7],
                "metadata": {"baseline_instance": [0.0, 0.0]},
            },
            {
                # Single-feature case still works and random equal-size masks are identical.
                "attributions": [0.4],
                "instance": [2.0],
                "prediction": 1,
                "prediction_proba": single_feature_model.predict_proba(np.array([[2.0]]))[0].tolist(),
                "metadata": {"baseline_instance": [0.0]},
            },
            {
                # Explanation-selected deletion is weaker than the random baseline, so score clips to 0.
                "attributions": [0.9, 0.1],
                "instance": [0.0, 5.0],
                "prediction": 1,
                "prediction_proba": model.predict_proba(np.array([[0.0, 5.0]]))[0].tolist(),
                "metadata": {"baseline_instance": [0.0, 0.0]},
            },
        ],
    }

    evaluator = CompletenessEvaluator(
        magnitude_threshold=1.0,
        min_features=1,
        random_trials=3,
        default_baseline=-1.0,
        fast_mode=False,
        random_state=5,
    )

    tie_scores = evaluator.evaluate(
        model,
        {**explanation_results, "current_index": 0},
        dataset=None,
        explainer=None,
    )
    zero_denom_scores = evaluator.evaluate(
        model,
        {**explanation_results, "current_index": 1},
        dataset=None,
        explainer=None,
    )
    missing_attr_scores = evaluator.evaluate(
        model,
        {**explanation_results, "current_index": 2},
        dataset=None,
        explainer=None,
    )
    single_scores = evaluator.evaluate(
        single_feature_model,
        {**explanation_results, "current_index": 3},
        dataset=None,
        explainer=None,
    )
    clipped_scores = evaluator.evaluate(
        model,
        {**explanation_results, "current_index": 4},
        dataset=None,
        explainer=None,
    )

    tied_instance = np.array([2.0, 1.0], dtype=float)
    tied_orig = explanation_results["explanations"][0]["prediction_proba"][1]
    tied_masked = tied_instance.copy()
    tied_masked[0] = -1.0
    expected_tie = abs(tied_orig - model.predict_proba(tied_masked.reshape(1, -1))[0, 1]) / (
        abs(tied_orig) + 1e-8
    )

    single_orig = explanation_results["explanations"][3]["prediction_proba"][1]
    single_new = single_feature_model.predict_proba(np.array([[0.0]]))[0, 1]
    expected_single = abs(single_orig - single_new) / (abs(single_orig) + 1e-8)

    selected_clipped = np.array([0.0, 5.0], dtype=float)
    selected_pred = explanation_results["explanations"][4]["prediction_proba"][1]
    selected_masked = np.array([0.0, 5.0], dtype=float)
    selected_drop = abs(selected_pred - model.predict_proba(selected_masked.reshape(1, -1))[0, 1]) / (
        abs(selected_pred) + 1e-8
    )
    random_masks = generate_random_masked_batch(
        selected_clipped,
        np.zeros_like(selected_clipped),
        n_trials=3,
        mask_size=1,
        rng=np.random.default_rng(9),
    )
    random_preds = model.predict_proba(random_masks)[:, 1]
    random_mean = np.mean(np.abs(selected_pred - random_preds) / (abs(selected_pred) + 1e-8))

    assert tie_scores["completeness_drop"] == pytest.approx(expected_tie, abs=1e-9)
    assert tie_scores["completeness_random_drop"] >= 0.0
    assert zero_denom_scores == {
        "completeness_drop": 0.0,
        "completeness_random_drop": 0.0,
        "completeness_score": 0.0,
    }
    assert missing_attr_scores == {
        "completeness_drop": 0.0,
        "completeness_random_drop": 0.0,
        "completeness_score": 0.0,
    }
    assert single_scores["completeness_drop"] == pytest.approx(expected_single, abs=1e-9)
    assert single_scores["completeness_random_drop"] == pytest.approx(expected_single, abs=1e-9)
    assert single_scores["completeness_score"] == pytest.approx(0.0, abs=1e-12)
    assert clipped_scores["completeness_drop"] == pytest.approx(selected_drop, abs=1e-9)
    assert clipped_scores["completeness_random_drop"] == pytest.approx(random_mean, abs=1e-9)
    assert clipped_scores["completeness_score"] == pytest.approx(0.0, abs=1e-12)


def test_completeness_fast_mode_preserves_original_expected_value_shortcut() -> None:
    evaluator = CompletenessEvaluator(fast_mode=True)
    explanation_results = {
        "method": "shap",
        "explanations": [
            {
                "attributions": [0.4, 0.2],
                "instance": [1.0, 2.0],
                "prediction": 1,
                "prediction_proba": [0.1, 0.9],
                "metadata": {"expected_value": [0.0, 0.2], "target": 1},
            }
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )

    output_diff = 0.9 - 0.2
    expected = 1.0 - abs((0.4 + 0.2) - output_diff) / abs(output_diff)

    assert scores["completeness_drop"] == pytest.approx(expected, abs=1e-9)
    assert scores["completeness_random_drop"] == pytest.approx(0.0)
    assert scores["completeness_score"] == pytest.approx(expected, abs=1e-9)
