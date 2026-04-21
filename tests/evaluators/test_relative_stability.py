from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.evaluators import (
    MetricRegistry,
    RelativeInputStabilityEvaluator,
    make_metric,
)


class DummyDataset:
    def __init__(self) -> None:
        self.X_train = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=float,
        )


class EchoExplainer:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def explain_instance(self, instance):
        arr = np.asarray(instance, dtype=float).reshape(-1)
        self.calls.append(arr.copy())
        return {
            "attributions": arr.tolist(),
            "instance": arr.tolist(),
        }


class ConstantExplainer:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=float).reshape(-1)
        self.calls: list[np.ndarray] = []

    def explain_instance(self, instance):
        arr = np.asarray(instance, dtype=float).reshape(-1)
        self.calls.append(arr.copy())
        return {
            "attributions": self.values.tolist(),
            "instance": arr.tolist(),
        }


class SequenceExplainer:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = [np.asarray(output, dtype=float).reshape(-1) for output in outputs]
        self.calls: list[np.ndarray] = []
        self._cursor = 0

    def explain_instance(self, instance):
        arr = np.asarray(instance, dtype=float).reshape(-1)
        self.calls.append(arr.copy())
        output = self.outputs[self._cursor]
        self._cursor += 1
        return {
            "attributions": output.tolist(),
            "instance": arr.tolist(),
        }


class FailingExplainer:
    def explain_instance(self, instance):
        raise RuntimeError("boom")


def test_relative_input_stability_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("relative_input_stability")

    assert isinstance(metric, RelativeInputStabilityEvaluator)
    assert metric.max_instances == 5
    assert metric.num_samples == 10
    assert metric.noise_scale == pytest.approx(0.01)
    assert metric.eps_min == pytest.approx(1.0e-03)
    assert metric.bounded is True
    assert metric.metric_names == ("relative_input_stability",)


def test_relative_input_stability_registry_entry_is_importable_from_metrics_config() -> None:
    registry = MetricRegistry()
    spec = registry.get("relative_input_stability")

    assert spec["module"] == "fed_perso_xai.evaluators.relative_stability"
    assert spec["class"] == "RelativeInputStabilityEvaluator"
    assert registry.is_available("relative_input_stability") is True


def test_relative_input_stability_echo_explainer_returns_expected_bounded_ratio() -> None:
    evaluator = RelativeInputStabilityEvaluator(
        max_instances=1,
        num_samples=4,
        noise_scale=0.05,
        random_state=7,
    )
    explanation_results = {
        "method": "shap",
        "explanations": [
            {
                "instance": [1.0, -2.0, 0.5],
                "attributions": [1.0, -2.0, 0.5],
            }
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=DummyDataset(),
        explainer=EchoExplainer(),
    )

    assert scores["relative_input_stability"] == pytest.approx(0.5, abs=1e-12)


def test_relative_input_stability_returns_zero_for_identical_zero_attributions() -> None:
    evaluator = RelativeInputStabilityEvaluator(
        max_instances=1,
        num_samples=3,
        noise_scale=0.05,
        random_state=3,
    )
    explanation_results = {
        "method": "shap",
        "explanations": [
            {
                "instance": [1.0, 2.0, 3.0],
                "attributions": [0.0, 0.0, 0.0],
            }
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=DummyDataset(),
        explainer=ConstantExplainer([0.0, 0.0, 0.0]),
    )

    assert scores["relative_input_stability"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("instance", "attributions"),
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([2.0], [2.0]),
    ],
)
def test_relative_input_stability_handles_zero_input_and_single_feature_without_nan(
    instance,
    attributions,
) -> None:
    evaluator = RelativeInputStabilityEvaluator(
        max_instances=1,
        num_samples=4,
        noise_scale=0.05,
        random_state=11,
    )
    explanation_results = {
        "method": "shap",
        "explanations": [
            {
                "instance": instance,
                "attributions": attributions,
            }
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=DummyDataset(),
        explainer=EchoExplainer(),
    )

    value = scores["relative_input_stability"]
    assert np.isfinite(value)
    assert value == pytest.approx(0.5, abs=1e-12)


def test_relative_input_stability_is_reproducible_for_fixed_seed() -> None:
    explanation_results = {
        "method": "shap",
        "explanations": [
            {
                "instance": [1.0, -2.0, 0.5],
                "attributions": [1.0, -2.0, 0.5],
            }
        ],
    }
    explainer_a = EchoExplainer()
    explainer_b = EchoExplainer()
    evaluator_a = RelativeInputStabilityEvaluator(
        max_instances=1,
        num_samples=5,
        noise_scale=0.05,
        random_state=19,
    )
    evaluator_b = RelativeInputStabilityEvaluator(
        max_instances=1,
        num_samples=5,
        noise_scale=0.05,
        random_state=19,
    )

    scores_a = evaluator_a.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=DummyDataset(),
        explainer=explainer_a,
    )
    scores_b = evaluator_b.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=DummyDataset(),
        explainer=explainer_b,
    )

    assert scores_a == pytest.approx(scores_b, abs=1e-12)
    assert len(explainer_a.calls) == len(explainer_b.calls) == 5
    for call_a, call_b in zip(explainer_a.calls, explainer_b.calls):
        np.testing.assert_allclose(call_a, call_b)


def test_relative_input_stability_discards_invalid_reruns_and_keeps_valid_samples() -> None:
    evaluator = RelativeInputStabilityEvaluator(
        max_instances=1,
        num_samples=3,
        noise_scale=0.05,
        random_state=23,
    )
    explanation_results = {
        "method": "shap",
        "explanations": [
            {
                "instance": [1.0, 2.0, 3.0],
                "attributions": [1.0, 2.0, 3.0],
            }
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=DummyDataset(),
        explainer=SequenceExplainer(
            outputs=[
                np.array([1.0, 2.0], dtype=float),
                np.array([np.nan, 2.0, 3.0], dtype=float),
                np.array([1.0, 2.0, 3.0], dtype=float),
            ]
        ),
    )

    assert scores["relative_input_stability"] == pytest.approx(0.0, abs=1e-12)


def test_relative_input_stability_returns_zero_when_all_reruns_fail() -> None:
    evaluator = RelativeInputStabilityEvaluator(
        max_instances=1,
        num_samples=3,
        noise_scale=0.05,
        random_state=5,
    )
    explanation_results = {
        "method": "shap",
        "explanations": [
            {
                "instance": [1.0, 2.0, 3.0],
                "attributions": [1.0, 2.0, 3.0],
            }
        ],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=DummyDataset(),
        explainer=FailingExplainer(),
    )

    assert scores["relative_input_stability"] == 0.0
