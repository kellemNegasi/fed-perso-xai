from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.evaluators import make_metric
from fed_perso_xai.evaluators.correctness import CorrectnessEvaluator


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


def test_correctness_metrics_are_registered_and_alias_is_instantiable() -> None:
    metric = make_metric("correctness")
    alias_metric = make_metric("output_completeness_deletion")

    assert isinstance(metric, CorrectnessEvaluator)
    assert metric.metric_key == "correctness"
    assert metric.fast_mode is False

    assert isinstance(alias_metric, CorrectnessEvaluator)
    assert alias_metric.metric_key == "output_completeness_deletion"


def test_correctness_feature_removal_matches_expected_drop() -> None:
    model = LinearProbModel(weights=[0.6, -0.4, 0.2])
    instance = np.array([1.0, -2.0, 0.5], dtype=float)
    proba = model.predict_proba(instance.reshape(1, -1))[0]

    explanation = {
        "attributions": [0.9, 0.2, 0.6],
        "instance": instance.tolist(),
        "metadata": {"baseline_instance": [0.0, 0.0, 0.0]},
        "prediction": 1,
        "prediction_proba": proba.tolist(),
    }
    explanation_results = {"method": "shap", "explanations": [explanation]}

    evaluator = CorrectnessEvaluator(
        removal_fraction=0.5,
        default_baseline=0.0,
        min_features=1,
        fast_mode=False,
    )
    scores = evaluator.evaluate(model, explanation_results, dataset=None, explainer=None)

    mask_indices = np.array([0, 2], dtype=int)
    orig_pred = proba[1]
    perturbed = instance.copy()
    perturbed[mask_indices] = 0.0
    new_pred = model.predict_proba(perturbed.reshape(1, -1))[0, 1]
    expected_drop = abs(orig_pred - new_pred) / (abs(orig_pred) + 1e-8)

    # Correctness should equal normalized probability drop after masking top features.
    assert scores["correctness"] == pytest.approx(expected_drop, abs=1e-9)


def test_correctness_uses_target_class_and_stable_tie_breaking() -> None:
    model = ThreeClassFeatureModel()
    instance = np.array([3.0, 1.0, 0.5], dtype=float)
    proba = model.predict_proba(instance.reshape(1, -1))[0]

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
                    "baseline_instance": [0.0, 0.0, 0.0],
                },
            }
        ],
    }

    evaluator = CorrectnessEvaluator(removal_fraction=1, fast_mode=False)
    scores = evaluator.evaluate(model, explanation_results, dataset=None, explainer=None)

    perturbed_feature0 = instance.copy()
    perturbed_feature0[0] = 0.0
    expected = abs(
        proba[0] - model.predict_proba(perturbed_feature0.reshape(1, -1))[0, 0]
    ) / (abs(proba[0]) + 1e-8)

    perturbed_feature1 = instance.copy()
    perturbed_feature1[1] = 0.0
    feature1_score = abs(
        proba[0] - model.predict_proba(perturbed_feature1.reshape(1, -1))[0, 0]
    ) / (abs(proba[0]) + 1e-8)

    assert scores["correctness"] == pytest.approx(expected, abs=1e-9)
    assert scores["correctness"] != pytest.approx(feature1_score, abs=1e-6)


def test_correctness_alias_reports_custom_metric_key() -> None:
    model = LinearProbModel(weights=[1.0])
    instance = np.array([2.0], dtype=float)
    proba = model.predict_proba(instance.reshape(1, -1))[0]
    explanation_results = {
        "method": "lime",
        "explanations": [
            {
                "attributions": [0.3],
                "instance": instance.tolist(),
                "prediction": 1,
                "prediction_proba": proba.tolist(),
                "metadata": {"baseline_instance": [0.0]},
            }
        ],
    }

    evaluator = CorrectnessEvaluator(
        metric_key="output_completeness_deletion",
        fast_mode=False,
    )
    scores = evaluator.evaluate(model, explanation_results, dataset=None, explainer=None)

    assert set(scores) == {"output_completeness_deletion"}
    assert 0.0 <= scores["output_completeness_deletion"] <= 1.0


def test_correctness_handles_zero_denominator_top_k_overflow_and_baseline_fallback() -> None:
    model = LinearProbModel(weights=[0.7, -0.2, 0.4])
    explanation_results = {
        "method": "integrated_gradients",
        "explanations": [
            {
                # All-zero importances still lead to a deterministic, stable top-k order.
                "attributions": [0.0, 0.0, 0.0],
                "instance": [1.0, 2.0, 3.0],
                "prediction": 1,
                "prediction_proba": model.predict_proba(
                    np.array([[1.0, 2.0, 3.0]])
                )[0].tolist(),
                "metadata": {"baseline_instance": [9.0]},  # wrong shape -> use default baseline
            },
            {
                "attributions": [0.4, 0.2],
                "instance": [5.0, -1.0],
                "prediction": 1,
                "prediction_proba": [1.0, 0.0],  # zero denominator should be skipped
                "metadata": {"baseline_instance": [0.0, 0.0]},
            },
        ],
    }

    evaluator = CorrectnessEvaluator(
        removal_fraction=10,
        default_baseline=-1.0,
        fast_mode=False,
    )

    first_scores = evaluator.evaluate(
        model,
        {**explanation_results, "current_index": 0},
        dataset=None,
        explainer=None,
    )
    second_scores = evaluator.evaluate(
        model,
        {**explanation_results, "current_index": 1},
        dataset=None,
        explainer=None,
    )

    fully_masked = np.array([-1.0, -1.0, -1.0], dtype=float)
    orig_pred = explanation_results["explanations"][0]["prediction_proba"][1]
    new_pred = model.predict_proba(fully_masked.reshape(1, -1))[0, 1]
    expected = abs(orig_pred - new_pred) / (abs(orig_pred) + 1e-8)

    assert first_scores["correctness"] == pytest.approx(expected, abs=1e-9)
    assert second_scores["correctness"] == pytest.approx(0.0)
