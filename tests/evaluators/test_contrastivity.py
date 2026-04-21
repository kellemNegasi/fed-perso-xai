from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.evaluators import ContrastivityEvaluator, make_metric


def _explanation(
    attributions,
    *,
    prediction=None,
    prediction_proba=None,
    target=None,
):
    explanation = {
        "instance": np.asarray(attributions, dtype=float).tolist(),
        "attributions": np.asarray(attributions, dtype=float).tolist(),
        "metadata": {},
    }
    if prediction is not None:
        explanation["prediction"] = prediction
    if prediction_proba is not None:
        explanation["prediction_proba"] = np.asarray(
            prediction_proba,
            dtype=float,
        ).tolist()
    if target is not None:
        explanation["metadata"]["target"] = int(target)
    return explanation


def test_contrastivity_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("contrastivity_ssim")

    assert isinstance(metric, ContrastivityEvaluator)
    assert metric.pairs_per_instance == 3
    assert metric.normalise is True
    assert metric.metric_names == ("contrastivity", "contrastivity_pairs")


def test_contrastivity_returns_expected_score_for_two_labels() -> None:
    evaluator = ContrastivityEvaluator(pairs_per_instance=1, random_state=0)
    explanation_results = {
        "method": "shap",
        "explanations": [
            _explanation([1.0, 0.0, 0.0], prediction=0),
            _explanation([0.0, 1.0, 0.0], prediction=1),
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )

    assert scores["contrastivity_pairs"] == pytest.approx(2.0, abs=1e-9)
    # Orthogonal importance vectors should yield max contrastivity (1 - SSIM ≈ 1).
    assert scores["contrastivity"] == pytest.approx(1.0, abs=1e-9)


def test_contrastivity_per_instance_anchors_on_target_explanation() -> None:
    evaluator = ContrastivityEvaluator(pairs_per_instance=2, random_state=0)
    payload = {
        "method": "shap",
        "current_index": 0,
        "explanations": [
            _explanation([1.0, 0.0, 0.0], prediction=0),
            _explanation([0.0, 1.0, 0.0], prediction=1),
            _explanation([0.0, 0.0, 1.0], prediction=1),
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=payload,
        dataset=None,
        explainer=None,
    )

    assert scores["contrastivity_pairs"] == pytest.approx(2.0, abs=1e-9)
    assert scores["contrastivity"] == pytest.approx(1.0, abs=1e-9)


def test_contrastivity_selects_off_class_references_by_predicted_label() -> None:
    evaluator = ContrastivityEvaluator(pairs_per_instance=4, random_state=0)
    payload = {
        "method": "lime",
        "current_index": 0,
        "explanations": [
            _explanation([1.0, 0.0, 0.0], prediction=0, target=1),
            _explanation([0.0, 1.0, 0.0], prediction=0, target=0),
            _explanation([1.0, 0.0, 0.0], prediction=1, target=1),
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=payload,
        dataset=None,
        explainer=None,
    )

    # The only admissible off-class reference is explanation 2 (prediction=1),
    # whose attribution vector matches the target explanation exactly.
    assert scores["contrastivity_pairs"] == pytest.approx(4.0, abs=1e-12)
    assert scores["contrastivity"] == pytest.approx(0.0, abs=1e-12)


def test_contrastivity_can_derive_labels_from_prediction_probabilities() -> None:
    evaluator = ContrastivityEvaluator(pairs_per_instance=1, random_state=2)
    payload = {
        "method": "causal_shap",
        "current_index": 0,
        "explanations": [
            _explanation([1.0, 0.0], prediction_proba=[0.8, 0.2]),
            _explanation([0.0, 1.0], prediction_proba=[0.1, 0.9]),
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=payload,
        dataset=None,
        explainer=None,
    )

    assert scores["contrastivity_pairs"] == pytest.approx(1.0, abs=1e-12)
    assert scores["contrastivity"] == pytest.approx(1.0, abs=1e-12)


def test_contrastivity_returns_zero_when_no_off_class_examples_are_available() -> None:
    evaluator = ContrastivityEvaluator(pairs_per_instance=3, random_state=0)
    payload = {
        "method": "integrated_gradients",
        "explanations": [
            _explanation([1.0, 0.0], prediction=1),
            _explanation([0.5, 0.5], prediction=1),
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=payload,
        dataset=None,
        explainer=None,
    )

    assert scores == {"contrastivity": 0.0, "contrastivity_pairs": 0.0}


def test_contrastivity_handles_single_feature_constant_vectors_and_identical_references() -> None:
    evaluator = ContrastivityEvaluator(pairs_per_instance=3, random_state=4)
    payload = {
        "method": "shap",
        "current_index": 0,
        "explanations": [
            _explanation([0.5], prediction=0),
            _explanation([0.5], prediction=1),
            _explanation([0.5], prediction=1),
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=payload,
        dataset=None,
        explainer=None,
    )

    assert scores["contrastivity_pairs"] == pytest.approx(3.0, abs=1e-12)
    assert scores["contrastivity"] == pytest.approx(0.0, abs=1e-12)


def test_contrastivity_skips_missing_or_nan_reference_explanations() -> None:
    evaluator = ContrastivityEvaluator(pairs_per_instance=4, random_state=1)
    payload = {
        "method": "shap",
        "current_index": 0,
        "explanations": [
            _explanation([1.0, 0.0], prediction=0),
            {"instance": [0.0, 1.0], "attributions": [np.nan, 1.0], "prediction": 1},
            {"instance": [0.0, 1.0], "prediction": 1},
            _explanation([0.0, 1.0], prediction=1),
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=payload,
        dataset=None,
        explainer=None,
    )

    assert scores["contrastivity_pairs"] == pytest.approx(4.0, abs=1e-12)
    assert scores["contrastivity"] == pytest.approx(1.0, abs=1e-12)


def test_contrastivity_sampling_is_deterministic_for_fixed_seed() -> None:
    payload = {
        "method": "shap",
        "current_index": 0,
        "explanations": [
            _explanation([1.0, 0.0], prediction=0),
            _explanation([1.0, 0.0], prediction=1),
            _explanation([0.0, 1.0], prediction=1),
        ],
    }
    evaluator_a = ContrastivityEvaluator(pairs_per_instance=6, random_state=7)
    evaluator_b = ContrastivityEvaluator(pairs_per_instance=6, random_state=7)

    scores_a = evaluator_a.evaluate(
        model=None,
        explanation_results=payload,
        dataset=None,
        explainer=None,
    )
    scores_b = evaluator_b.evaluate(
        model=None,
        explanation_results=payload,
        dataset=None,
        explainer=None,
    )

    assert scores_a == pytest.approx(scores_b, abs=1e-12)
