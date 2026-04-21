from __future__ import annotations

import numpy as np
import pytest

from fed_perso_xai.evaluators import ConfidenceEvaluator, make_metric


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
        self.y_train = np.array([0, 1, 0], dtype=int)


class CloneTrackingExplainer:
    init_records: list["CloneTrackingExplainer"] = []
    fit_records: list[tuple[np.ndarray, np.ndarray | None]] = []
    explain_records: list[tuple[int | None, np.ndarray]] = []

    def __init__(self, config, model, dataset):
        self.config = dict(config)
        self.model = model
        self.dataset = dataset
        self.random_state = None
        type(self).init_records.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.init_records = []
        cls.fit_records = []
        cls.explain_records = []

    def fit(self, X, y=None) -> None:
        X_arr = np.asarray(X, dtype=float)
        y_arr = None if y is None else np.asarray(y)
        type(self).fit_records.append((X_arr.copy(), None if y_arr is None else y_arr.copy()))

    def explain_instance(self, instance):
        arr = np.asarray(instance, dtype=float).reshape(-1)
        type(self).explain_records.append((self.random_state, arr.copy()))
        offset = 0.0 if self.random_state is None else float(self.random_state % 17) / 100.0
        return {
            "attributions": (arr + offset).tolist(),
            "instance": arr.tolist(),
        }


class DeterministicCloneExplainer(CloneTrackingExplainer):
    def explain_instance(self, instance):
        arr = np.asarray(instance, dtype=float).reshape(-1)
        type(self).explain_records.append((self.random_state, arr.copy()))
        return {
            "attributions": [0.25] * arr.size,
            "instance": arr.tolist(),
        }


class SeededNoiseExplainer(CloneTrackingExplainer):
    def explain_instance(self, instance):
        arr = np.asarray(instance, dtype=float).reshape(-1)
        type(self).explain_records.append((self.random_state, arr.copy()))
        rng = np.random.default_rng(self.random_state)
        noise = rng.normal(loc=0.0, scale=0.1, size=arr.size)
        return {
            "attributions": (arr + noise).tolist(),
            "instance": arr.tolist(),
        }


class NaNRerunConfidenceEvaluator(ConfidenceEvaluator):
    def __init__(self, reruns: list[np.ndarray | None], **kwargs) -> None:
        super().__init__(**kwargs)
        self._reruns = list(reruns)

    def _rerun_explainer(self, metric_input, instance, rng):
        if not self._reruns:
            return None
        return self._reruns.pop(0)


def _base_results():
    explanations = [
        {"instance": [1.0, 2.0, 3.0], "attributions": [1.0, 2.0, 3.0]},
        {"instance": [4.0], "attributions": [0.25]},
    ]
    return {"method": "shap", "explanations": explanations}


def test_confidence_metric_is_registered_and_instantiable() -> None:
    metric = make_metric("confidence")

    assert isinstance(metric, ConfidenceEvaluator)
    assert metric.n_resamples == 8
    assert metric.ci_percentile == pytest.approx(95.0)
    assert metric.max_instances == 10
    assert metric.noise_scale == pytest.approx(0.005)
    assert metric.random_baseline is True
    assert metric.metric_names == ("confidence",)


def test_confidence_from_samples_computes_expected_interval_widths_and_weighted_score() -> None:
    evaluator = ConfidenceEvaluator(ci_percentile=50.0)
    samples = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=float,
    )

    per_feature, aggregate = evaluator._confidence_from_samples(samples)

    expected_per_feature = np.array(
        [
            1.0 - (2.0 / (3.0 + 2.0 + 1e-8)),
            1.0 - (2.0 / (4.0 + 2.0 + 1e-8)),
        ],
        dtype=float,
    )
    expected_aggregate = float(np.dot(expected_per_feature, np.array([3.0, 4.0])) / 7.0)
    np.testing.assert_allclose(per_feature, expected_per_feature, atol=1e-12)
    assert aggregate == pytest.approx(expected_aggregate, abs=1e-12)


def test_confidence_falls_back_to_unweighted_mean_when_attribution_mass_is_zero() -> None:
    evaluator = ConfidenceEvaluator()
    samples = np.zeros((4, 3), dtype=float)

    per_feature, aggregate = evaluator._confidence_from_samples(samples)

    np.testing.assert_allclose(per_feature, np.ones(3, dtype=float), atol=1e-12)
    assert aggregate == pytest.approx(1.0, abs=1e-12)


def test_confidence_reruns_clone_with_same_config_and_training_data() -> None:
    dataset = DummyDataset()
    model = object()
    explainer = CloneTrackingExplainer(
        config={"type": "shap", "experiment": {"explanation": {"random_state": 13}}},
        model=model,
        dataset=dataset,
    )
    CloneTrackingExplainer.reset()
    evaluator = ConfidenceEvaluator(n_resamples=4, noise_scale=0.0, random_state=11)
    explanation_results = {
        "method": "shap",
        "explanations": [{"instance": [1.0, 2.0, 3.0], "attributions": [1.0, 2.0, 3.0]}],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=dataset,
        explainer=explainer,
    )

    assert 0.0 <= scores["confidence"] <= 1.0
    assert len(CloneTrackingExplainer.init_records) == 3
    assert len(CloneTrackingExplainer.fit_records) == 3
    assert len(CloneTrackingExplainer.explain_records) == 3
    for clone in CloneTrackingExplainer.init_records:
        assert clone.config == explainer.config
        assert clone.model is model
        assert clone.dataset is dataset
    for X_train, y_train in CloneTrackingExplainer.fit_records:
        np.testing.assert_allclose(X_train, dataset.X_train)
        assert y_train is not None
        np.testing.assert_array_equal(y_train, dataset.y_train)
    metadata = explanation_results["explanations"][0]["metadata"]
    assert np.all(np.isfinite(metadata["confidence_per_feature"]))
    assert len(metadata["confidence_per_feature"]) == 3


def test_confidence_returns_one_for_identical_deterministic_reruns() -> None:
    dataset = DummyDataset()
    explainer = DeterministicCloneExplainer(
        config={"type": "shap"},
        model=object(),
        dataset=dataset,
    )
    DeterministicCloneExplainer.reset()
    evaluator = ConfidenceEvaluator(n_resamples=5, noise_scale=0.0, random_state=3)
    explanation_results = {
        "method": "shap",
        "explanations": [{"instance": [1.0, 2.0, 3.0], "attributions": [0.25, 0.25, 0.25]}],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=dataset,
        explainer=explainer,
    )

    assert scores["confidence"] == pytest.approx(1.0, abs=1e-12)


def test_confidence_handles_single_feature_inputs() -> None:
    dataset = DummyDataset()
    explainer = DeterministicCloneExplainer(
        config={"type": "shap"},
        model=object(),
        dataset=dataset,
    )
    DeterministicCloneExplainer.reset()
    evaluator = ConfidenceEvaluator(n_resamples=3, noise_scale=0.0, random_state=9)
    payload = _base_results()
    payload["current_index"] = 1

    scores = evaluator.evaluate(
        model=None,
        explanation_results=payload,
        dataset=dataset,
        explainer=explainer,
    )

    assert scores["confidence"] == pytest.approx(1.0, abs=1e-12)


def test_confidence_discards_nan_reruns_and_keeps_valid_samples() -> None:
    evaluator = NaNRerunConfidenceEvaluator(
        reruns=[
            np.array([np.nan, 1.0, 1.0], dtype=float),
            np.array([1.0, 2.0, 3.0], dtype=float),
        ],
        n_resamples=3,
        noise_scale=0.0,
    )
    explanation_results = {
        "method": "shap",
        "explanations": [{"instance": [1.0, 2.0, 3.0], "attributions": [1.0, 2.0, 3.0]}],
    }

    scores = evaluator.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=DummyDataset(),
        explainer=object(),
    )

    assert scores["confidence"] == pytest.approx(1.0, abs=1e-12)


def test_confidence_is_reproducible_for_fixed_seed() -> None:
    dataset = DummyDataset()
    explainer_a = SeededNoiseExplainer(config={"type": "shap"}, model=object(), dataset=dataset)
    explainer_b = SeededNoiseExplainer(config={"type": "shap"}, model=object(), dataset=dataset)
    SeededNoiseExplainer.reset()
    evaluator_a = ConfidenceEvaluator(n_resamples=4, noise_scale=0.0, random_state=21)
    evaluator_b = ConfidenceEvaluator(n_resamples=4, noise_scale=0.0, random_state=21)
    explanation_results = {
        "method": "shap",
        "explanations": [{"instance": [1.0, -2.0, 0.5], "attributions": [1.0, -2.0, 0.5]}],
    }

    scores_a = evaluator_a.evaluate(
        model=None,
        explanation_results=explanation_results,
        dataset=dataset,
        explainer=explainer_a,
    )
    scores_b = evaluator_b.evaluate(
        model=None,
        explanation_results={
            "method": "shap",
            "explanations": [{"instance": [1.0, -2.0, 0.5], "attributions": [1.0, -2.0, 0.5]}],
        },
        dataset=dataset,
        explainer=explainer_b,
    )

    assert scores_a == pytest.approx(scores_b, abs=1e-12)
