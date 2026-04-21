"""
Helpers for resolving the explanation class used by evaluator-side metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _coerce_class_index(value: Any) -> int | None:
    """Return an integer class index when metadata encodes one safely."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        rounded = int(round(float(value)))
        if abs(float(value) - rounded) < 1.0e-9:
            return rounded
    return None


def _probability_row(probabilities: Any) -> np.ndarray | None:
    """Coerce probability outputs into a 1D row without assuming batch shape."""
    if probabilities is None:
        return None
    try:
        arr = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        row = arr
    else:
        row = np.asarray(arr[0], dtype=float).reshape(-1)
    row = row.reshape(-1)
    if row.size == 0 or not np.all(np.isfinite(row)):
        return None
    return row


def _prediction_probabilities(
    explanation: dict[str, Any],
    *,
    model: Any = None,
    instance: Any = None,
) -> np.ndarray | None:
    """Read probabilities from the explanation first, then the model if needed."""
    proba = _probability_row(explanation.get("prediction_proba"))
    if proba is not None:
        return proba

    if model is None or instance is None or not hasattr(model, "predict_proba"):
        return None

    try:
        row = np.asarray(instance, dtype=float).reshape(1, -1)
        return _probability_row(model.predict_proba(row))
    except Exception:
        return None


def _binary_fallback_class(
    explanation: dict[str, Any],
    *,
    model: Any = None,
) -> int | None:
    """Return the canonical positive-class index when binary classification is evident."""
    proba = _probability_row(explanation.get("prediction_proba"))
    if proba is not None and proba.size == 2:
        return 1

    classes = getattr(model, "classes_", None) if model is not None else None
    if classes is not None:
        try:
            if len(classes) == 2:
                return 1
        except TypeError:
            return None

    return None


def resolve_explained_class(
    explanation: dict[str, Any],
    model: Any = None,
    instance: Any = None,
) -> int | None:
    """
    Resolve which class should be used for metric evaluation.

    Priority:
    1. ``metadata.explained_class`` when explicitly set by the explainer.
    2. The predicted class derived from probability outputs.
    3. The binary positive-class fallback (class index ``1``).

    Notes
    -----
    ``metadata.true_label`` is intentionally ignored here. Ground-truth labels
    describe the supervised target, not the class whose prediction score an
    explanation metric should track. When no classification signal is available
    (for example, regression outputs or invalid metadata), the resolver returns
    ``None`` so evaluators keep their scalar-score semantics unchanged.
    """
    proba = _prediction_probabilities(explanation, model=model, instance=instance)

    metadata = explanation.get("metadata")
    if isinstance(metadata, dict):
        explicit = _coerce_class_index(metadata.get("explained_class"))
        if explicit is not None and explicit >= 0:
            if proba is not None:
                if explicit < proba.size:
                    return explicit
            else:
                classes = getattr(model, "classes_", None) if model is not None else None
                if classes is None or explicit < len(classes):
                    return explicit

    if proba is not None and proba.size > 0:
        return int(np.argmax(proba))

    fallback = _binary_fallback_class(explanation, model=model)
    if fallback is not None:
        return fallback

    return None
