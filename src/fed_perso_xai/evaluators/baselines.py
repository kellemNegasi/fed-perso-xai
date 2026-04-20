"""
Shared baseline-selection and feature-scale helpers for explanation metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np


def baseline_vector(
    explanation: dict[str, Any],
    instance: np.ndarray,
    *,
    default_baseline: float = 0.0,
    dataset: Any | None = None,
    logger: logging.Logger | None = None,
    log_prefix: str = "Evaluator",
) -> np.ndarray:
    """
    Return explainer-provided baseline if present, otherwise derive one from dataset/defaults.

    Several Perso-XAI metrics follow this same order:
      1. ``metadata["baseline_instance"]`` if shape-compatible.
      2. Dataset mean over ``X_train`` when available.
      3. Constant fallback value.
    """
    metadata = explanation.get("metadata") or {}
    baseline = metadata.get("baseline_instance")
    if baseline is not None:
        base_arr = np.asarray(baseline, dtype=float).reshape(-1)
        if base_arr.shape == instance.shape:
            return base_arr

    if dataset is not None:
        X_train = getattr(dataset, "X_train", None)
        if X_train is not None:
            X_arr = np.asarray(X_train, dtype=float)
            if X_arr.ndim >= 2 and X_arr.shape[1] == instance.shape[0]:
                return np.mean(X_arr, axis=0).reshape(-1)

    if logger is not None:
        logger.debug("%s using default baseline %.5f", log_prefix, default_baseline)
    return np.full_like(instance, float(default_baseline), dtype=float)


def dataset_feature_std(
    dataset: Any | None,
    example_explanation: dict[str, Any] | None = None,
) -> Optional[np.ndarray]:
    """Return flattened feature standard deviations derived from the dataset."""
    if dataset is None:
        return None
    X_train = getattr(dataset, "X_train", None)
    if X_train is None:
        return None
    X_arr = np.asarray(X_train)
    if X_arr.ndim == 1:
        return np.std(X_arr, axis=0).reshape(1)
    if X_arr.ndim >= 2:
        if X_arr.ndim > 2:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)
        std = np.std(X_arr, axis=0)
        std[std == 0.0] = 1e-6
        return std
    return None


def feature_scale(instance: np.ndarray) -> np.ndarray:
    """Feature-wise scale used for perturbation noise magnitudes."""
    return np.maximum(np.abs(instance), 1e-3)
