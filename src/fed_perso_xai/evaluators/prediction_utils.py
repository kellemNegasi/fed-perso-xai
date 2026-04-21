"""
Shared helpers for metric-side prediction access and target selection.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _model_has_score_outputs(
    model: Any,
    *,
    prefer_probability: bool = True,
) -> bool:
    """Return whether the model can emit score/probability outputs beyond hard labels."""
    if model is None:
        return False
    if prefer_probability and hasattr(model, "predict_proba"):
        return True
    return hasattr(model, "decision_function")


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


def resolve_scalar_prediction_score(
    explanation: dict[str, Any] | None = None,
    *,
    model: Any = None,
    instance: Any = None,
    target_class: int | None = None,
    prefer_probability: bool = True,
) -> Optional[float]:
    """
    Resolve the scalar prediction value tracked by deletion-style metrics.

    The canonical quantity follows the original evaluator intent: use the
    target-class probability/score for classification when available, otherwise
    use the scalar model prediction. Cached explanation probabilities are reused
    when present. Hard labels from ``prediction`` are only used as a fallback
    when neither the explanation nor the model can provide a richer score.
    """
    if explanation is not None and prefer_probability:
        proba = explanation.get("prediction_proba")
        if proba is not None:
            value = prediction_value_from_probabilities(
                proba,
                target_class=target_class,
            )
            if value is not None:
                return float(value)

    should_query_model = (
        model is not None
        and instance is not None
        and (
            _model_has_score_outputs(
                model,
                prefer_probability=prefer_probability,
            )
            or explanation is None
        )
    )
    if should_query_model:
        try:
            return float(
                model_prediction(
                    model,
                    np.asarray(instance, dtype=float),
                    target_class=target_class,
                    prefer_probability=prefer_probability,
                )
            )
        except Exception:
            pass

    if explanation is None:
        return None
    value = extract_prediction_value(
        explanation,
        target_class=target_class,
        prefer_probability=prefer_probability,
    )
    if value is None:
        return None
    return float(value)


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
