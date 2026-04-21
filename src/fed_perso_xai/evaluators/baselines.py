"""
Shared baseline-resolution and feature-scale helpers for explanation metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np


BASELINE_STRATEGY_EXPLAINER_ONLY = "prefer_explainer"
BASELINE_STRATEGY_EXPLAINER_THEN_DATASET_MEAN = "prefer_explainer_then_dataset_mean"
BaselineResolutionStrategy = Literal[
    "prefer_explainer",
    "prefer_explainer_then_dataset_mean",
]


def resolve_baseline_vector(
    explanation: dict[str, Any],
    instance: np.ndarray,
    *,
    strategy: BaselineResolutionStrategy = BASELINE_STRATEGY_EXPLAINER_THEN_DATASET_MEAN,
    default_baseline: float = 0.0,
    dataset: Any | None = None,
    logger: logging.Logger | None = None,
    log_prefix: str = "Evaluator",
) -> np.ndarray:
    """
    Resolve the concrete baseline vector used for masking and perturbation metrics.

    Resolution order is centralized here so evaluators do not quietly invent their
    own fallback semantics:

      1. ``metadata["baseline_instance"]`` when it is numeric, finite, and shape-compatible.
      2. Dataset mean over ``dataset.X_train`` when the selected strategy allows it.
      3. A documented constant fallback value as the last resort.

    Strategies
    ----------
    ``prefer_explainer``
        Mirrors the original Perso-XAI evaluators that only accepted an
        explainer-provided baseline vector and otherwise fell back to a scalar
        constant.
    ``prefer_explainer_then_dataset_mean``
        Used by metrics such as infidelity that intentionally allow a
        dataset-derived reference vector before falling back to a scalar.
    """
    instance_arr = np.asarray(instance, dtype=float).reshape(-1)
    metadata = explanation.get("metadata") or {}

    baseline = _coerce_candidate_baseline(
        metadata.get("baseline_instance"),
        instance_arr,
        logger=logger,
        log_prefix=log_prefix,
        source_name='metadata["baseline_instance"]',
    )
    if baseline is not None:
        return baseline

    if strategy == BASELINE_STRATEGY_EXPLAINER_THEN_DATASET_MEAN:
        dataset_baseline = _dataset_mean_baseline(
            dataset,
            instance_arr,
            logger=logger,
            log_prefix=log_prefix,
        )
        if dataset_baseline is not None:
            return dataset_baseline
    elif strategy != BASELINE_STRATEGY_EXPLAINER_ONLY:
        raise ValueError(f"Unsupported baseline resolution strategy: {strategy!r}")

    fallback_value = _coerce_default_baseline(default_baseline)
    if logger is not None:
        logger.debug("%s using constant baseline %.5f", log_prefix, fallback_value)
    return np.full_like(instance_arr, fallback_value, dtype=float)


def baseline_vector(
    explanation: dict[str, Any],
    instance: np.ndarray,
    *,
    strategy: BaselineResolutionStrategy = BASELINE_STRATEGY_EXPLAINER_THEN_DATASET_MEAN,
    default_baseline: float = 0.0,
    dataset: Any | None = None,
    logger: logging.Logger | None = None,
    log_prefix: str = "Evaluator",
) -> np.ndarray:
    """
    Backwards-compatible alias for ``resolve_baseline_vector``.

    New evaluator code should prefer the explicit helper name so baseline
    semantics are easy to audit at call sites.
    """
    return resolve_baseline_vector(
        explanation,
        instance,
        strategy=strategy,
        default_baseline=default_baseline,
        dataset=dataset,
        logger=logger,
        log_prefix=log_prefix,
    )


def _coerce_candidate_baseline(
    candidate: Any,
    instance: np.ndarray,
    *,
    logger: logging.Logger | None,
    log_prefix: str,
    source_name: str,
) -> np.ndarray | None:
    if candidate is None:
        return None

    try:
        baseline = np.asarray(candidate, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        if logger is not None:
            logger.debug("%s ignoring non-numeric %s", log_prefix, source_name)
        return None

    if baseline.shape != instance.shape:
        if logger is not None:
            logger.debug(
                "%s ignoring %s with shape %s (expected %s)",
                log_prefix,
                source_name,
                baseline.shape,
                instance.shape,
            )
        return None

    if not np.all(np.isfinite(baseline)):
        if logger is not None:
            logger.debug("%s ignoring non-finite %s", log_prefix, source_name)
        return None

    return baseline.astype(float, copy=True)


def _dataset_mean_baseline(
    dataset: Any | None,
    instance: np.ndarray,
    *,
    logger: logging.Logger | None,
    log_prefix: str,
) -> np.ndarray | None:
    if dataset is None:
        return None

    X_train = getattr(dataset, "X_train", None)
    if X_train is None:
        if logger is not None:
            logger.debug("%s cannot derive dataset baseline: dataset.X_train is missing", log_prefix)
        return None

    try:
        X_arr = np.asarray(X_train, dtype=float)
    except (TypeError, ValueError):
        if logger is not None:
            logger.debug("%s cannot derive dataset baseline: dataset.X_train is non-numeric", log_prefix)
        return None

    if X_arr.size == 0:
        if logger is not None:
            logger.debug("%s cannot derive dataset baseline from an empty dataset", log_prefix)
        return None

    if X_arr.ndim == 1:
        mean = np.asarray([np.mean(X_arr, dtype=float)], dtype=float)
    else:
        if X_arr.shape[0] == 0:
            if logger is not None:
                logger.debug("%s cannot derive dataset baseline from an empty dataset", log_prefix)
            return None
        if X_arr.ndim > 2:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)
        if X_arr.shape[1] != instance.shape[0]:
            if logger is not None:
                logger.debug(
                    "%s cannot derive dataset baseline: feature shape %s does not match instance shape %s",
                    log_prefix,
                    X_arr.shape[1:],
                    instance.shape,
                )
            return None
        mean = np.mean(X_arr, axis=0).reshape(-1)

    if mean.shape != instance.shape:
        if logger is not None:
            logger.debug(
                "%s cannot derive dataset baseline: mean shape %s does not match instance shape %s",
                log_prefix,
                mean.shape,
                instance.shape,
            )
        return None

    if not np.all(np.isfinite(mean)):
        if logger is not None:
            logger.debug("%s ignoring non-finite dataset mean baseline", log_prefix)
        return None

    return mean.astype(float, copy=True)


def _coerce_default_baseline(default_baseline: float) -> float:
    try:
        value = float(default_baseline)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"default_baseline must be a finite numeric scalar. Got {default_baseline!r}."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"default_baseline must be a finite numeric scalar. Got {default_baseline!r}."
        )
    return value


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
