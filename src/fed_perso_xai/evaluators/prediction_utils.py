"""
Shared helpers for metric-side prediction access and target selection.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def extract_prediction_value(
    explanation: dict[str, Any],
    *,
    target_class: int | None = None,
    prefer_probability: bool = True,
) -> Optional[float]:
    """
    Scalar value to track under feature removal.

    Prefer class probability if available; otherwise use the raw prediction.
    """
    if prefer_probability:
        proba = explanation.get("prediction_proba")
        if proba is not None:
            return prediction_value_from_probabilities(proba, target_class=target_class)

    prediction = explanation.get("prediction")
    if prediction is None:
        return None
    arr = np.asarray(prediction).ravel()
    if arr.size == 0:
        return None
    try:
        return float(arr[0])
    except Exception:
        return None


def prediction_value_from_probabilities(
    probabilities: Any,
    *,
    target_class: int | None = None,
) -> Optional[float]:
    """Select one scalar probability from a binary or multiclass output."""
    proba_arr = np.asarray(probabilities, dtype=float).ravel()
    if proba_arr.size == 0:
        return None
    if target_class is not None and 0 <= int(target_class) < proba_arr.size:
        return float(proba_arr[int(target_class)])
    if proba_arr.size == 2:
        return float(proba_arr[1])
    return float(np.max(proba_arr))


def prediction_label(explanation: dict[str, Any]) -> Any:
    """Derive the predicted label from explanation (uses prediction or proba)."""
    prediction = explanation.get("prediction")
    if prediction is not None:
        if isinstance(prediction, (str, bytes)):
            return prediction
        pred_arr = np.asarray(prediction)
        if pred_arr.ndim == 0:
            value = pred_arr.item()
        else:
            if pred_arr.size == 0:
                value = None
            else:
                value = pred_arr.ravel()[0]
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return value
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            rounded = int(round(float(value)))
            if abs(float(value) - rounded) < 1e-6:
                return rounded
            return None

    proba = explanation.get("prediction_proba")
    if proba is None:
        return None
    proba_arr = np.asarray(proba).ravel()
    if proba_arr.size == 0:
        return None
    return int(np.argmax(proba_arr))


def model_prediction(
    model: Any,
    instance: np.ndarray,
    *,
    target_class: int | None = None,
    prefer_probability: bool = True,
) -> float:
    """
    Compute scalar prediction for a perturbed instance, aligned with explanation helpers:
    prefer class probability if available; otherwise use decision_function or raw prediction.
    """
    batch = np.asarray(instance, dtype=float).reshape(1, -1)
    values = model_predictions(
        model,
        batch,
        target_class=target_class,
        prefer_probability=prefer_probability,
    )
    if values.size == 0:
        raise ValueError("Model prediction helper returned empty output.")
    return float(values[0])


def model_predictions(
    model: Any,
    instances: np.ndarray,
    *,
    target_class: int | None = None,
    prefer_probability: bool = True,
) -> np.ndarray:
    """
    Return scalar outputs for a batch of instances.

    We first prefer ``predict_proba`` to stay aligned with the original evaluator
    semantics. If probabilities are unavailable we fall back to ``decision_function``
    and finally to ``predict``.
    """
    batch = np.asarray(instances, dtype=float)

    if prefer_probability and hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(batch))
        if proba.ndim == 1:
            return proba.astype(float)
        if target_class is not None and 0 <= int(target_class) < proba.shape[1]:
            return proba[:, int(target_class)].astype(float)
        if proba.shape[1] == 2:
            return proba[:, 1].astype(float)
        return np.max(proba, axis=1).astype(float)

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(batch))
        if decision.ndim == 1:
            return decision.astype(float)
        if decision.ndim == 2:
            if target_class is not None and 0 <= int(target_class) < decision.shape[1]:
                return decision[:, int(target_class)].astype(float)
            return np.max(decision, axis=1).astype(float)

    if hasattr(model, "predict"):
        preds = np.asarray(model.predict(batch)).reshape(batch.shape[0])
        return preds.astype(float)

    raise AttributeError("Model must expose predict(), decision_function(), or predict_proba().")
