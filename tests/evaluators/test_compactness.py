from __future__ import annotations

import pytest

from fed_perso_xai.evaluators import make_metric
from fed_perso_xai.evaluators.compactness import CompactnessEvaluator


def test_compactness_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("compactness_size")

    assert isinstance(metric, CompactnessEvaluator)
    assert metric.zero_tolerance == pytest.approx(1.0e-08)


def test_compactness_scores_match_expected_distribution() -> None:
    evaluator = CompactnessEvaluator(zero_tolerance=1e-9)
    explanation_results = {
        "method": "shap",
        "explanations": [
            {
                "attributions": [0.5, 0.0, -0.5, 0.0],
                "instance": [1.0, 2.0, 3.0, 4.0],
            }
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )

    # Two of four features are non-zero -> sparsity 0.5.
    assert scores["compactness_sparsity"] == pytest.approx(0.5)
    # Total attribution mass captured by top 5 features equals 1.0.
    assert scores["compactness_top5_coverage"] == pytest.approx(1.0)
    # Total attribution mass captured by top 10 features also equals 1.0.
    assert scores["compactness_top10_coverage"] == pytest.approx(1.0)
    # Effective feature count normalization yields 2/3 for this distribution.
    assert scores["compactness_effective_features"] == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_compactness_top_k_coverage_matches_expected_mass() -> None:
    evaluator = CompactnessEvaluator()
    explanation_results = {
        "method": "lime",
        "explanations": [
            {
                "attributions": [5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )

    assert scores["compactness_sparsity"] == pytest.approx(0.0)
    assert scores["compactness_top5_coverage"] == pytest.approx(15.0 / 22.0)
    assert scores["compactness_top10_coverage"] == pytest.approx(20.0 / 22.0)


def test_compactness_handles_zero_uniform_dominant_and_threshold_boundary_cases() -> None:
    evaluator = CompactnessEvaluator(zero_tolerance=1.0e-03)
    explanation_results = {
        "method": "integrated_gradients",
        "explanations": [
            {"attributions": [0.0, 0.0, 0.0, 0.0]},
            {"attributions": [1.0, 0.0, 0.0, 0.0]},
            {"attributions": [1.0] * 12},
            {"attributions": [1.0e-03, 2.0e-03, 0.0]},
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
    boundary_scores = evaluator.evaluate(
        model=None,
        explanation_results={**explanation_results, "current_index": 3},
        dataset=None,
        explainer=None,
    )

    assert zero_scores["compactness_sparsity"] == pytest.approx(1.0)
    assert zero_scores["compactness_top5_coverage"] == pytest.approx(0.0)
    assert zero_scores["compactness_top10_coverage"] == pytest.approx(0.0)
    assert zero_scores["compactness_effective_features"] == pytest.approx(0.0)

    assert dominant_scores["compactness_sparsity"] == pytest.approx(0.75)
    assert dominant_scores["compactness_top5_coverage"] == pytest.approx(1.0)
    assert dominant_scores["compactness_top10_coverage"] == pytest.approx(1.0)
    assert dominant_scores["compactness_effective_features"] == pytest.approx(1.0)

    assert uniform_scores["compactness_sparsity"] == pytest.approx(0.0)
    assert uniform_scores["compactness_top5_coverage"] == pytest.approx(5.0 / 12.0)
    assert uniform_scores["compactness_top10_coverage"] == pytest.approx(10.0 / 12.0)
    assert uniform_scores["compactness_effective_features"] == pytest.approx(0.0)

    # Magnitudes exactly on the tolerance boundary are treated as "near-zero".
    assert boundary_scores["compactness_sparsity"] == pytest.approx(2.0 / 3.0)
