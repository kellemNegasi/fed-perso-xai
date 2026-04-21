from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fed_perso_xai.evaluators.completeness as completeness_module
import fed_perso_xai.evaluators.correctness as correctness_module
import fed_perso_xai.evaluators.infidelity as infidelity_module
import fed_perso_xai.evaluators.monotonicity as monotonicity_module
import fed_perso_xai.evaluators.non_sensitivity as non_sensitivity_module
from fed_perso_xai.evaluators import resolve_baseline_vector
from fed_perso_xai.evaluators.baselines import (
    BASELINE_STRATEGY_EXPLAINER_ONLY,
    BASELINE_STRATEGY_EXPLAINER_THEN_DATASET_MEAN,
)


def _explanation(instance, *, baseline=None):
    metadata = {}
    if baseline is not None:
        metadata["baseline_instance"] = baseline
    return {
        "instance": np.asarray(instance, dtype=float).tolist(),
        "attributions": np.ones(len(instance), dtype=float).tolist(),
        "metadata": metadata,
    }


def test_resolve_baseline_vector_prefers_valid_metadata() -> None:
    instance = np.array([1.0, 2.0, 3.0], dtype=float)
    explanation = _explanation(instance, baseline=[0.0, 0.0, 0.0])
    dataset = SimpleNamespace(X_train=np.array([[5.0, 5.0, 5.0]], dtype=float))

    baseline = resolve_baseline_vector(
        explanation,
        instance,
        strategy=BASELINE_STRATEGY_EXPLAINER_THEN_DATASET_MEAN,
        dataset=dataset,
        default_baseline=-1.0,
    )

    np.testing.assert_allclose(baseline, np.zeros_like(instance))


def test_resolve_baseline_vector_uses_dataset_mean_when_metadata_is_invalid() -> None:
    instance = np.array([1.0, 2.0, 3.0], dtype=float)
    explanation = _explanation(instance, baseline=[9.0])
    dataset = SimpleNamespace(
        X_train=np.array(
            [
                [1.0, 3.0, 5.0],
                [3.0, 5.0, 7.0],
            ],
            dtype=float,
        )
    )

    baseline = resolve_baseline_vector(
        explanation,
        instance,
        strategy=BASELINE_STRATEGY_EXPLAINER_THEN_DATASET_MEAN,
        dataset=dataset,
        default_baseline=-1.0,
    )

    np.testing.assert_allclose(baseline, np.array([2.0, 4.0, 6.0]))


def test_resolve_baseline_vector_falls_back_safely_for_non_finite_metadata_and_empty_dataset() -> None:
    instance = np.array([1.0, 2.0, 3.0], dtype=float)
    explanation = _explanation(instance, baseline=[0.0, np.nan, 0.0])
    dataset = SimpleNamespace(X_train=np.empty((0, 3), dtype=float))

    baseline = resolve_baseline_vector(
        explanation,
        instance,
        strategy=BASELINE_STRATEGY_EXPLAINER_THEN_DATASET_MEAN,
        dataset=dataset,
        default_baseline=-2.5,
    )

    np.testing.assert_allclose(baseline, np.full_like(instance, -2.5))


def test_resolve_baseline_vector_rejects_invalid_default_baseline() -> None:
    instance = np.array([1.0, 2.0], dtype=float)
    explanation = _explanation(instance)

    with pytest.raises(ValueError, match="default_baseline"):
        resolve_baseline_vector(
            explanation,
            instance,
            strategy=BASELINE_STRATEGY_EXPLAINER_ONLY,
            default_baseline=float("nan"),
        )


def test_replacement_metrics_route_through_shared_baseline_resolver(monkeypatch) -> None:
    calls: list[tuple[str, str, bool]] = []

    def _record(explanation, instance, **kwargs):
        calls.append(
            (
                kwargs["log_prefix"],
                kwargs["strategy"],
                kwargs.get("dataset") is not None,
            )
        )
        return np.zeros_like(np.asarray(instance, dtype=float).reshape(-1))

    monkeypatch.setattr(correctness_module, "resolve_baseline_vector", _record)
    monkeypatch.setattr(completeness_module, "resolve_baseline_vector", _record)
    monkeypatch.setattr(non_sensitivity_module, "resolve_baseline_vector", _record)
    monkeypatch.setattr(infidelity_module, "resolve_baseline_vector", _record)
    monkeypatch.setattr(monotonicity_module, "resolve_baseline_vector", _record)

    explanation = _explanation([1.0, 2.0, 3.0], baseline=[0.0, 0.0, 0.0])
    instance = np.array([1.0, 2.0, 3.0], dtype=float)
    dataset = SimpleNamespace(X_train=np.ones((2, 3), dtype=float))

    correctness_module.CorrectnessEvaluator()._baseline_vector(explanation, instance)
    completeness_module.CompletenessEvaluator()._baseline_vector(explanation, instance)
    non_sensitivity_module.NonSensitivityEvaluator()._baseline_vector(explanation, instance)
    infidelity_module.InfidelityEvaluator()._baseline_vector(
        explanation,
        instance,
        dataset=dataset,
    )
    monotonicity_module.MonotonicityEvaluator()._baseline_vector(explanation, instance)

    assert calls == [
        ("CorrectnessEvaluator", BASELINE_STRATEGY_EXPLAINER_ONLY, False),
        ("CompletenessEvaluator", BASELINE_STRATEGY_EXPLAINER_ONLY, False),
        ("NonSensitivityEvaluator", BASELINE_STRATEGY_EXPLAINER_ONLY, False),
        (
            "InfidelityEvaluator",
            BASELINE_STRATEGY_EXPLAINER_THEN_DATASET_MEAN,
            True,
        ),
        ("MonotonicityEvaluator", BASELINE_STRATEGY_EXPLAINER_ONLY, False),
    ]
