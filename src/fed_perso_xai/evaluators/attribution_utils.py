"""
Shared attribution extraction / formatting helpers for explanation metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np


FEATURE_ATTRIBUTION_FIELD_CANDIDATES = (
    "feature_importance",
    "feature_importances",
    "attributions",
    "importance",
)

FEATURE_METHOD_KEYS = {
    "shap",
    "lime",
    "integrated_gradients",
    "integratedgradients",
    "causal_shap",
    "causalshap",
}


def extract_attribution_vector(
    explanation: dict[str, Any],
    *,
    candidates: Sequence[str] = FEATURE_ATTRIBUTION_FIELD_CANDIDATES,
    logger: logging.Logger | None = None,
    log_prefix: str = "Evaluator",
) -> Optional[np.ndarray]:
    """
    Extract attribution vector from explanation or metadata.

    Looks in both the root of the explanation and inside metadata so we can support
    multiple explainer schemas.
    """
    metadata = explanation.get("metadata") or {}
    for container in (explanation, metadata):
        for key in candidates:
            vec = container.get(key)
            arr = coerce_attribution_vector(vec, logger=logger, log_prefix=log_prefix)
            if arr is not None:
                return arr
    return None


def coerce_attribution_vector(
    values: Any,
    *,
    logger: logging.Logger | None = None,
    log_prefix: str = "Evaluator",
) -> Optional[np.ndarray]:
    """Coerce importance values into a 1-D float numpy array."""
    if values is None:
        if logger is not None:
            logger.debug("%s missing importance vector in explanation", log_prefix)
        return None
    if isinstance(values, np.ndarray):
        arr = values
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        arr = np.asarray(values)
    else:
        if logger is not None:
            logger.debug("%s could not coerce attribution values to an array", log_prefix)
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(float)


def extract_instance_vector(explanation: dict[str, Any]) -> Optional[np.ndarray]:
    """Extract the instance represented by an explanation as a 1-D numpy vector."""
    candidate = (
        explanation.get("instance")
        or (explanation.get("metadata") or {}).get("instance")
        or explanation.get("input")
    )
    if candidate is None:
        return None
    arr = np.asarray(candidate, dtype=float).reshape(-1)
    return arr.copy()


def prepare_attributions(
    vec: np.ndarray,
    *,
    abs_attributions: bool = False,
    normalise: bool = False,
    normalise_mode: str = "max_abs",
) -> np.ndarray:
    """
    Prepare attributions for metrics that work on magnitude or normalised weights.

    The default ``"max_abs"`` path mirrors the Infidelity/Monotonicity helpers,
    while ``"l1"`` is used by Contrastivity so attribution vectors remain comparable
    across instances.
    """
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if abs_attributions:
        arr = np.abs(arr)
    if not normalise:
        return arr

    if normalise_mode == "l1":
        norm = np.sum(np.abs(arr))
        if norm < 1e-12:
            return arr
        return arr / norm

    denom = np.max(np.abs(arr))
    if denom > 0:
        arr = arr / denom
    return arr
