from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.evaluators import ContinuityEvaluator, make_metric


class DummyDataset:
    def __init__(self):
        self.X_train = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )


class EchoExplainer:
    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)
        self.calls: list[np.ndarray] = []

    def explain_instance(self, instance):
        arr = np.asarray(instance, dtype=float).reshape(-1)
        self.calls.append(arr.copy())
        return {
            "attributions": (self.scale * arr).tolist(),
            "instance": arr.tolist(),
        }


class SequenceExplainer:
    def __init__(self, outputs):
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


def _base_results():
    explanations = [
        {"instance": [1.0, 2.0, 3.0], "attributions": [1.0, 2.0, 3.0]},
        {"instance": [0.5, 1.5, 2.5], "attributions": [1.0, 2.0, 3.0]},
    ]
    return {"method": "shap", "explanations": explanations}


def test_continuity_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("continuity_stability")

    assert isinstance(metric, ContinuityEvaluator)
    assert metric.max_instances == 5
    assert metric.noise_scale == pytest.approx(0.01)
    assert metric.metric_names == ("continuity_stability",)


def test_continuity_batch_zero_noise_yields_expected_average_and_reruns_explainer() -> None:
    explanation_results = _base_results()
    evaluator = ContinuityEvaluator(max_instances=2, noise_scale=0.0)
    dataset = DummyDataset()
    explainer = SequenceExplainer(
        outputs=[
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 1.0],
        ]
    )

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=dataset,
        explainer=explainer,
    )

    assert scores["continuity_stability"] == pytest.approx(0.5, abs=1e-12)
    assert len(explainer.calls) == 2
    np.testing.assert_allclose(explainer.calls[0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(explainer.calls[1], np.array([0.5, 1.5, 2.5]))


def test_continuity_per_instance_uses_current_index() -> None:
    explanation_results = _base_results()
    evaluator = ContinuityEvaluator(max_instances=2, noise_scale=0.0)
    dataset = DummyDataset()
    explainer = EchoExplainer(scale=1.0)

    payload = dict(explanation_results)
    payload["current_index"] = 1
    scores = evaluator.evaluate(
        model=None,
        explanation_results=payload,
        dataset=dataset,
        explainer=explainer,
    )

    assert scores["continuity_stability"] == pytest.approx(1.0, abs=1e-12)
    assert len(explainer.calls) == 1
    np.testing.assert_allclose(explainer.calls[0], np.array([0.5, 1.5, 2.5]))


def test_continuity_noise_is_deterministic_for_fixed_seed() -> None:
    explanation_results = {
        "method": "shap",
        "explanations": [
            {
                "instance": np.array([1.0, -2.0, 0.5], dtype=float),
                "attributions": np.array([1.0, -2.0, 0.5], dtype=float),
            }
        ],
    }
    dataset = DummyDataset()

    explainer_a = EchoExplainer(scale=1.0)
    explainer_b = EchoExplainer(scale=1.0)
    evaluator_a = ContinuityEvaluator(max_instances=1, noise_scale=0.05, random_state=7)
    evaluator_b = ContinuityEvaluator(max_instances=1, noise_scale=0.05, random_state=7)

    scores_a = evaluator_a.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=dataset,
        explainer=explainer_a,
    )
    scores_b = evaluator_b.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=dataset,
        explainer=explainer_b,
    )

    assert scores_a == pytest.approx(scores_b, abs=1e-12)
    assert len(explainer_a.calls) == len(explainer_b.calls) == 1
    np.testing.assert_allclose(explainer_a.calls[0], explainer_b.calls[0])


@pytest.mark.parametrize(
    ("instance", "attributions"),
    [
        ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]),
        ([1.0], [0.3]),
    ],
)
def test_continuity_returns_zero_for_degenerate_correlation_cases(instance, attributions) -> None:
    evaluator = ContinuityEvaluator(max_instances=1, noise_scale=0.0)
    explainer = EchoExplainer(scale=1.0)
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
        explainer=explainer,
    )

    assert scores["continuity_stability"] == 0.0
