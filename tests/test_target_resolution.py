from __future__ import annotations

import numpy as np

from fed_perso_xai.utils.target_resolution import resolve_explained_class


class BinaryProbModel:
    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        positive = 0.2 + 0.4 * X_arr[:, 0]
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


class BinaryClassesOnlyModel:
    classes_ = np.asarray([0, 1])


def test_resolve_explained_class_prefers_explicit_metadata() -> None:
    explanation = {
        "prediction_proba": [0.1, 0.9],
        "metadata": {"explained_class": 0, "true_label": 1},
    }

    assert resolve_explained_class(explanation) == 0


def test_resolve_explained_class_uses_probability_fallback_without_metadata() -> None:
    model = BinaryProbModel()
    explanation = {"metadata": {}}

    assert resolve_explained_class(explanation, model=model, instance=np.array([2.0, 0.0])) == 1


def test_resolve_explained_class_uses_binary_positive_class_fallback() -> None:
    explanation = {"metadata": {"true_label": 0}}

    assert resolve_explained_class(explanation, model=BinaryClassesOnlyModel()) == 1


def test_resolve_explained_class_returns_none_for_regression_like_payloads() -> None:
    explanation = {
        "prediction": 0.37,
        "metadata": {"explained_class": "not-a-class", "true_label": 1},
    }

    assert resolve_explained_class(explanation) is None
