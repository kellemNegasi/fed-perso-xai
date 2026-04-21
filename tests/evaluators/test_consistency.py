from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.evaluators import ConsistencyEvaluator, make_metric
from fed_perso_xai.evaluators.consistency import _top_n_sign


def _explanation(
    attributions,
    *,
    prediction=None,
    prediction_proba=None,
):
    explanation = {
        "instance": np.asarray(attributions, dtype=float).reshape(-1).tolist(),
        "attributions": np.asarray(attributions, dtype=float).reshape(-1).tolist(),
        "metadata": {},
    }
    if prediction is not None:
        explanation["prediction"] = prediction
    if prediction_proba is not None:
        explanation["prediction_proba"] = np.asarray(
            prediction_proba,
            dtype=float,
        ).reshape(-1).tolist()
    return explanation


def test_consistency_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("consistency_local")

    assert isinstance(metric, ConsistencyEvaluator)
    assert metric.discretise_kwargs == {"n": 10}
    assert metric.metric_names == ("consistency",)
    assert metric.requires_full_batch is True


def test_default_top_n_sign_discretiser_uses_magnitude_and_stable_tie_breaks() -> None:
    token = _top_n_sign(np.array([0.5, -0.5, 0.2], dtype=float), n=2)

    # Features 0 and 1 tie in magnitude, so the lower index wins first.
    assert token == ((0, 1), (1, -1))


def test_consistency_matches_same_explanation_groups_against_predicted_labels() -> None:
    evaluator = ConsistencyEvaluator(discretise_kwargs={"n": 2})
    explanation_results = {
        "method": "shap",
        "explanations": [
            _explanation([1.0, 0.0], prediction=0),
            _explanation([2.0, 0.0], prediction=0),
            _explanation([3.0, 0.0], prediction=1),
            _explanation([0.0, 1.0], prediction=1),
        ],
    }

    batch_scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )
    per_instance_scores = evaluator.evaluate(
        model=None,
        explanation_results={**explanation_results, "current_index": 2},
        dataset=None,
        explainer=None,
    )

    assert batch_scores["consistency"] == pytest.approx(0.25, abs=1e-12)
    assert per_instance_scores["consistency"] == pytest.approx(0.0, abs=1e-12)


def test_consistency_can_derive_labels_from_prediction_probabilities() -> None:
    evaluator = ConsistencyEvaluator(discretise_kwargs={"n": 1})
    explanation_results = {
        "method": "lime",
        "current_index": 0,
        "explanations": [
            _explanation([0.2, 0.0], prediction_proba=[0.9, 0.1]),
            _explanation([0.4, 0.0], prediction_proba=[0.8, 0.2]),
            _explanation([0.8, 0.0], prediction_proba=[0.2, 0.8]),
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )

    assert scores["consistency"] == pytest.approx(0.5, abs=1e-12)


def test_consistency_supports_dotted_path_discretisers_and_hashable_token_canonicalisation() -> None:
    evaluator = ConsistencyEvaluator(
        discretise_func="fed_perso_xai.evaluators.consistency._top_n_sign",
        discretise_kwargs={"n": 1},
    )
    explanation_results = {
        "method": "causal_shap",
        "current_index": 1,
        "explanations": [
            _explanation([1.0, 0.0], prediction=1),
            _explanation([2.0, 0.0], prediction=1),
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )

    assert scores["consistency"] == pytest.approx(1.0, abs=1e-12)


def test_consistency_returns_zero_for_singletons_distinct_tokens_and_single_instance_batches() -> None:
    evaluator = ConsistencyEvaluator(discretise_kwargs={"n": 1})

    distinct_scores = evaluator.evaluate(
        model=None,
        explanation_results={
            "method": "integrated_gradients",
            "explanations": [
                _explanation([1.0, 0.0], prediction=0),
                _explanation([0.0, 1.0], prediction=1),
            ],
        },
        dataset=None,
        explainer=None,
    )
    singleton_scores = evaluator.evaluate(
        model=None,
        explanation_results={
            "method": "integrated_gradients",
            "explanations": [
                _explanation([1.0], prediction=0),
            ],
        },
        dataset=None,
        explainer=None,
    )

    assert distinct_scores["consistency"] == 0.0
    assert singleton_scores["consistency"] == 0.0


def test_consistency_clamps_n_for_single_feature_inputs_and_preserves_expected_score() -> None:
    evaluator = ConsistencyEvaluator(discretise_kwargs={"n": 10})
    explanation_results = {
        "method": "shap",
        "explanations": [
            _explanation([0.5], prediction=0),
            _explanation([1.5], prediction=0),
            _explanation([2.5], prediction=1),
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )

    assert scores["consistency"] == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_consistency_skips_nan_and_missing_explanations_safely() -> None:
    evaluator = ConsistencyEvaluator(discretise_kwargs={"n": 1})
    explanation_results = {
        "method": "shap",
        "explanations": [
            _explanation([1.0, 0.0], prediction=0),
            {"instance": [1.0, 0.0], "attributions": [np.nan, 0.0], "prediction": 0},
            {"instance": [1.0, 0.0], "prediction": 0},
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=None,
        explainer=None,
    )

    assert scores["consistency"] == 0.0
