from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.evaluators import make_metric
from fed_perso_xai.evaluators.covariate_complexity import CovariateComplexityEvaluator


def _normalized_entropy(importance: np.ndarray) -> float:
    abs_imp = np.abs(importance)
    prob = abs_imp / np.sum(abs_imp)
    safe_prob = np.clip(prob, 1e-12, 1.0)
    entropy = -np.sum(safe_prob * np.log2(safe_prob))
    max_entropy = np.log2(len(prob)) if len(prob) > 1 else 0.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def test_covariate_complexity_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("covariate_complexity")

    assert isinstance(metric, CovariateComplexityEvaluator)
    assert metric.metric_key == "covariate_complexity"
    assert metric.regularity_key == "covariate_regularity"


def test_covariate_complexity_matches_entropy_expectation() -> None:
    explanation_results = {
        "method": "shap",
        "explanations": [
            {"attributions": [0.5, 0.25, 0.25]},
            {"attributions": [1.0, 0.0, 0.0]},
        ],
    }

    evaluator = CovariateComplexityEvaluator()
    scores = evaluator.evaluate(model=None, explanation_results=explanation_results, dataset=None, explainer=None)

    norm1 = _normalized_entropy(np.array([0.5, 0.25, 0.25], dtype=float))
    norm2 = _normalized_entropy(np.array([1.0, 0.0, 0.0], dtype=float))
    expected_complexity = np.mean([norm1, norm2])
    expected_regularity = np.mean([1.0 - norm1, 1.0 - norm2])

    # Average normalized entropy must equal mean of the per-instance values.
    assert scores["covariate_complexity"] == pytest.approx(expected_complexity, abs=1e-9)
    # Regularity is defined as 1 - complexity on a per-instance basis, so averages match too.
    assert scores["covariate_regularity"] == pytest.approx(expected_regularity, abs=1e-9)


def test_covariate_complexity_handles_edge_cases_and_metadata_fallback() -> None:
    evaluator = CovariateComplexityEvaluator()
    explanation_results = {
        "method": "integrated_gradients",
        "explanations": [
            {"attributions": [0.0, 0.0, 0.0, 0.0]},
            {"attributions": [1.0, 0.0, 0.0, 0.0]},
            {"metadata": {"feature_importance": [1.0, 1.0, 1.0, 1.0]}},
            {"attributions": [1.0e-15, 2.0e-15]},
            {"attributions": [7.0]},
            {"attributions": [1.0, 1.0]},
        ],
    }

    zero_scores = evaluator.evaluate(
        model=None,
        explanation_results={**explanation_results, "current_index": 0},
        dataset=None,
        explainer=None,
    )
    dominant_scores = evaluator.evaluate(
        model=None,
        explanation_results={**explanation_results, "current_index": 1},
        dataset=None,
        explainer=None,
    )
    uniform_scores = evaluator.evaluate(
        model=None,
        explanation_results={**explanation_results, "current_index": 2},
        dataset=None,
        explainer=None,
    )
    tiny_scores = evaluator.evaluate(
        model=None,
        explanation_results={**explanation_results, "current_index": 3},
        dataset=None,
        explainer=None,
    )
    single_feature_scores = evaluator.evaluate(
        model=None,
        explanation_results={**explanation_results, "current_index": 4},
        dataset=None,
        explainer=None,
    )
    two_feature_uniform_scores = evaluator.evaluate(
        model=None,
        explanation_results={**explanation_results, "current_index": 5},
        dataset=None,
        explainer=None,
    )

    assert zero_scores["covariate_complexity"] == pytest.approx(0.0)
    assert zero_scores["covariate_regularity"] == pytest.approx(1.0)

    assert dominant_scores["covariate_complexity"] == pytest.approx(
        _normalized_entropy(np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
        abs=1e-12,
    )
    assert dominant_scores["covariate_regularity"] == pytest.approx(
        1.0 - _normalized_entropy(np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
        abs=1e-12,
    )

    assert uniform_scores["covariate_complexity"] == pytest.approx(1.0)
    assert uniform_scores["covariate_regularity"] == pytest.approx(0.0)

    assert tiny_scores["covariate_complexity"] == pytest.approx(
        _normalized_entropy(np.array([1.0e-15, 2.0e-15], dtype=float)),
        abs=1e-9,
    )
    assert tiny_scores["covariate_regularity"] == pytest.approx(
        1.0 - _normalized_entropy(np.array([1.0e-15, 2.0e-15], dtype=float)),
        abs=1e-9,
    )

    assert single_feature_scores["covariate_complexity"] == pytest.approx(0.0)
    assert single_feature_scores["covariate_regularity"] == pytest.approx(1.0)

    assert two_feature_uniform_scores["covariate_complexity"] == pytest.approx(1.0)
    assert two_feature_uniform_scores["covariate_regularity"] == pytest.approx(0.0)
