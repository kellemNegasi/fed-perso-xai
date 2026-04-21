"""
Non-sensitivity evaluator mirrored after Nguyen & Rodriguez Martinez (2020).

Checks whether features assigned near-zero attribution truly have negligible
influence by perturbing them and measuring the resulting change in model output.
Adapted from the Quantus `NonSensitivity` metric (https://github.com/understandable-machine-intelligence-lab/Quantus).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.target_resolution import resolve_explained_class
from .attribution_utils import (
    FEATURE_METHOD_KEYS,
    extract_attribution_vector,
    extract_instance_vector,
)
from .base_metric import MetricCapabilities, MetricInput
from .baselines import baseline_vector
from .perturbation import chunk_indices, mask_feature_indices
from .prediction_utils import (
    model_prediction,
    resolve_scalar_prediction_score,
)


_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


@dataclass
class _NonSensitivityContext:
    """Cached payload with everything needed to score an explanation."""

    importance: np.ndarray
    instance: np.ndarray
    baseline: np.ndarray
    zero_indices: np.ndarray
    original_prediction: float
    target_class: int | None


class NonSensitivityEvaluator(MetricCapabilities):
    """
    Flag situations where a feature receives (near) zero attribution yet materially
    changes the prediction when perturbed.

    Metrics
    -------
    non_sensitivity_violation_fraction
        Fraction of zero-attribution features that caused prediction changes larger
        than ``delta_tolerance`` when perturbed (lower is better).
    non_sensitivity_safe_fraction
        Complementary fraction of zero-attribution features whose perturbations
        stayed below the tolerance threshold.
    non_sensitivity_zero_features
        Average count of features flagged as zero attribution per explanation.
    non_sensitivity_delta_mean
        Mean absolute prediction change observed while probing zero-attribution
        features, indicating violation severity.

    Parameters
    ----------
    zero_threshold : float, optional
        Absolute attribution magnitude treated as zero importance.
    delta_tolerance : float, optional
        Maximum prediction delta tolerated when perturbing zero-attribution features.
        Deltas above this tolerance are counted as violations.
    features_per_step : int, optional
        Number of zero-attribution features to perturb simultaneously. When
        greater than one, all features in the group inherit the same verdict.
    default_baseline : float, optional
        Value used to fill masked features when explanations do not provide a
        ``baseline_instance`` in the metadata.
    """

    per_instance = True
    requires_full_batch = False
    metric_names = (
        "non_sensitivity_violation_fraction",
        "non_sensitivity_safe_fraction",
        "non_sensitivity_zero_features",
        "non_sensitivity_delta_mean",
    )
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        zero_threshold: float = 1e-5,
        delta_tolerance: float = 1e-4,
        features_per_step: int = 1,
        default_baseline: float = 0.0,
        cache_context: bool = True,
    ) -> None:
        self.zero_threshold = float(max(0.0, zero_threshold))
        self.delta_tolerance = float(max(0.0, delta_tolerance))
        self.features_per_step = max(1, int(features_per_step))
        self.default_baseline = float(default_baseline)
        self.cache_context = bool(cache_context)
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: Any,
        explanation_results: Dict[str, Any],
        dataset: Any | None = None,
        explainer: Any | None = None,
        cache: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        metric_input = MetricInput.from_results(
            model=model,
            explanation_results=explanation_results,
            dataset=dataset,
            explainer=explainer,
            cache=cache,
        )
        return self._evaluate(metric_input)

    def _evaluate(self, metric_input: MetricInput) -> Dict[str, float]:
        if metric_input.method not in self.supported_methods:
            return self._empty_result()

        explanations = metric_input.explanations
        if not explanations:
            return self._empty_result()

        context_cache: Optional[Dict[int, _NonSensitivityContext]] = (
            metric_input.cache_bucket("non_sensitivity_context") if self.cache_context else None
        )

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return self._empty_result()
            metrics = self._metrics_for_explanation(
                metric_input.model,
                explanations[idx],
                context_cache=context_cache,
            )
            return metrics or self._empty_result()

        violation_rates: List[float] = []
        safe_rates: List[float] = []
        zero_counts: List[float] = []
        delta_means: List[float] = []

        for _, explanation in metric_input.iter_explanations():
            metrics = self._metrics_for_explanation(
                metric_input.model,
                explanation,
                context_cache=context_cache,
            )
            if not metrics:
                continue
            violation_rates.append(metrics["non_sensitivity_violation_fraction"])
            safe_rates.append(metrics["non_sensitivity_safe_fraction"])
            zero_counts.append(metrics["non_sensitivity_zero_features"])
            delta_means.append(metrics["non_sensitivity_delta_mean"])

        if not violation_rates:
            return self._empty_result()

        return {
            "non_sensitivity_violation_fraction": float(np.mean(violation_rates)),
            "non_sensitivity_safe_fraction": float(np.mean(safe_rates)),
            "non_sensitivity_zero_features": float(np.mean(zero_counts)),
            "non_sensitivity_delta_mean": float(np.mean(delta_means)),
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _empty_result(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.metric_names}

    def _metrics_for_explanation(
        self,
        model: Any,
        explanation: Dict[str, Any],
        *,
        context_cache: Optional[Dict[int, _NonSensitivityContext]] = None,
    ) -> Optional[Dict[str, float]]:
        context = self._prepare_context(model, explanation, context_cache)
        if context is None:
            return None

        if context.zero_indices.size == 0:
            return {
                "non_sensitivity_violation_fraction": 0.0,
                "non_sensitivity_safe_fraction": 0.0,
                "non_sensitivity_zero_features": 0.0,
                "non_sensitivity_delta_mean": 0.0,
            }

        deltas: List[float] = []
        violations = 0
        safe = 0

        for group in chunk_indices(context.zero_indices, features_per_step=self.features_per_step):
            perturbed = mask_feature_indices(
                context.instance,
                group,
                context.baseline,
            )
            try:
                perturbed_pred = self._model_prediction(
                    model,
                    perturbed,
                    target_class=context.target_class,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("NonSensitivityEvaluator failed to score perturbation: %s", exc)
                continue
            delta = float(abs(context.original_prediction - perturbed_pred))
            deltas.append(delta)
            if delta > self.delta_tolerance:
                violations += len(group)
            else:
                safe += len(group)

        total = violations + safe
        if total == 0:
            return {
                "non_sensitivity_violation_fraction": 0.0,
                "non_sensitivity_safe_fraction": 0.0,
                "non_sensitivity_zero_features": float(context.zero_indices.size),
                "non_sensitivity_delta_mean": 0.0,
            }

        violation_fraction = float(violations / total)
        safe_fraction = float(safe / total)
        avg_delta = float(np.mean(deltas)) if deltas else 0.0

        return {
            "non_sensitivity_violation_fraction": violation_fraction,
            "non_sensitivity_safe_fraction": safe_fraction,
            "non_sensitivity_zero_features": float(context.zero_indices.size),
            "non_sensitivity_delta_mean": avg_delta,
        }

    def _prepare_context(
        self,
        model: Any,
        explanation: Dict[str, Any],
        context_cache: Optional[Dict[int, _NonSensitivityContext]],
    ) -> Optional[_NonSensitivityContext]:
        cache_key = id(explanation)
        if context_cache is not None and cache_key in context_cache:
            return context_cache[cache_key]

        importance = self._importance_vector(explanation)
        if importance is None or importance.size == 0:
            return None

        instance = self._extract_instance(explanation)
        if instance is None:
            return None
        if instance.size != importance.size:
            self.logger.debug(
                "NonSensitivityEvaluator length mismatch: instance=%s, importance=%s",
                instance.size,
                importance.size,
            )
            return None

        baseline = self._baseline_vector(explanation, instance)
        zero_indices = np.flatnonzero(np.abs(importance) <= self.zero_threshold)
        target_class = self._target_class(explanation, model, instance)
        original_prediction = self._prediction_value(
            explanation,
            model=model,
            instance=instance,
            target_class=target_class,
        )
        if original_prediction is None:
            return None

        context = _NonSensitivityContext(
            importance=importance,
            instance=instance,
            baseline=baseline,
            zero_indices=zero_indices,
            original_prediction=float(original_prediction),
            target_class=target_class,
        )
        if context_cache is not None:
            context_cache[cache_key] = context
        return context

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        arr = extract_attribution_vector(
            explanation,
            logger=self.logger,
            log_prefix="NonSensitivityEvaluator",
        )
        if arr is not None:
            return arr
        self.logger.debug("NonSensitivityEvaluator missing attribution vector.")
        return None

    def _extract_instance(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        return extract_instance_vector(explanation)

    def _baseline_vector(self, explanation: Dict[str, Any], instance: np.ndarray) -> np.ndarray:
        # The original non-sensitivity evaluator only consumed an explainer-provided
        # baseline_instance or the configured scalar fallback; keep that behavior
        # rather than deriving a dataset-mean baseline here.
        baseline = baseline_vector(
            explanation,
            instance,
            default_baseline=self.default_baseline,
            dataset=None,
            logger=self.logger,
            log_prefix="NonSensitivityEvaluator",
        )
        return baseline

    def _target_class(
        self,
        explanation: Dict[str, Any],
        model: Any,
        instance: np.ndarray,
    ) -> int | None:
        """
        Resolve the class whose score should be tracked under perturbation.

        This uses the shared resolver so ground-truth labels are never reused as
        evaluator targets.
        """
        return resolve_explained_class(explanation, model=model, instance=instance)

    def _prediction_value(
        self,
        explanation: Dict[str, Any],
        *,
        model: Any | None = None,
        instance: np.ndarray | None = None,
        target_class: int | None,
    ) -> Optional[float]:
        value = resolve_scalar_prediction_score(
            explanation,
            model=model,
            instance=instance,
            target_class=target_class,
            prefer_probability=True,
        )
        if value is not None:
            return float(value)
        return None

    def _model_prediction(
        self,
        model: Any,
        instance: np.ndarray,
        *,
        target_class: int | None,
    ) -> float:
        return float(
            model_prediction(
                model,
                instance,
                target_class=target_class,
                prefer_probability=True,
            )
        )
