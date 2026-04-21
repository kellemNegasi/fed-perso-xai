"""
Correctness (faithfulness) metric for local tabular explanations.

Implements a feature-removal test: the more the model prediction changes after
masking the most important features (according to the explanation), the more
"correct" the explanation is with respect to the black-box.
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
from .baselines import BASELINE_STRATEGY_EXPLAINER_ONLY, resolve_baseline_vector
from .base_metric import MetricCapabilities, MetricInput
from .perturbation import mask_feature_indices, top_k_mask_indices
from .prediction_utils import (
    model_prediction,
    resolve_scalar_prediction_score,
)


_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


@dataclass
class _CorrectnessContext:
    """Cached payload with everything needed to score an explanation."""

    importance: np.ndarray
    instance: np.ndarray
    baseline: Optional[np.ndarray]
    top_indices: np.ndarray
    original_prediction: float
    target_class: int | None


class CorrectnessEvaluator(MetricCapabilities):
    """
    Computes a correctness score via a feature-removal test.

    Parameters
    ----------
    metric_key : str, optional
        Dictionary key used for the deletion score. Defaults to ``"correctness"``
        and also supports the YAML alias ``"output_completeness_deletion"``.
    removal_fraction : float | int
        Fraction of top-ranked features (by absolute importance) to mask when given
        as a float in [0, 1]. If an integer is provided, it is interpreted as the
        absolute number of top features to delete (use 1 for single-feature deletion).
    default_baseline : float
        Value used to replace masked features when no baseline vector is provided
        in the explanation metadata (via ``baseline_instance``).
    min_features : int
        Minimum number of features to mask, regardless of ``removal_fraction``.

    Notes
    -----
    The scalar quantity tracked under deletion is the same in both execution
    modes: target-class probability/score for classification when available,
    otherwise the scalar regression prediction. ``fast_mode`` is therefore a
    performance optimization only; it must not change the prediction semantics.
    """

    per_instance = True
    requires_full_batch = False
    metric_names = ("correctness",)
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        metric_key: str = "correctness",
        removal_fraction: float = 0.1,
        default_baseline: float = 0.0,
        min_features: int = 1,
        fast_mode: bool = True,
        cache_context: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        metric_key : str, optional
            Dictionary key used for the deletion score. This keeps the original
            ``correctness`` name while also honoring the existing
            ``output_completeness_deletion`` YAML alias.
        removal_fraction : float | int, optional
            Fraction (or absolute count) of top-ranked features to mask when computing
            the deletion score.
        default_baseline : float, optional
            Value substituted for masked features when an explanation does not provide
            its own ``baseline_instance`` metadata.
        min_features : int, optional
            Enforce a minimum number of masked features even if ``removal_fraction``
            would suggest fewer.
        fast_mode : bool, optional
            Reuse cached context and explanation-provided probability scores when
            available, while preserving the same scalar target-score semantics as
            the full deletion path.
        """
        if isinstance(removal_fraction, bool):
            # avoid treating booleans as integers; coerce to float fraction
            removal_fraction = float(removal_fraction)

        if isinstance(removal_fraction, (int, np.integer)):
            self._removal_mode = "count"
            self._removal_count = max(1, int(removal_fraction))
            self.removal_fraction = None
        else:
            self._removal_mode = "fraction"
            self.removal_fraction = float(np.clip(float(removal_fraction), 0.0, 1.0))
            self._removal_count = None

        self.metric_key = metric_key or "correctness"
        self.metric_names = (self.metric_key,)
        self.default_baseline = float(default_baseline)
        self.min_features = max(1, int(min_features))
        self.fast_mode = bool(fast_mode)
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
        """
        Evaluate correctness for SHAP/LIME/IG/Causal SHAP explanations.

        Parameters
        ----------
        model : Any
            Trained model associated with the explanations (must expose predict /
            predict_proba used during feature masking).
        explanation_results : Dict[str, Any]
            Output of ``BaseExplainer.explain_dataset`` (or compatible structure
            containing ``method`` and ``explanations`` entries).
        dataset : Any | None, optional
            Dataset object (unused currently but accepted for interface parity).
        explainer : Any | None, optional
            Explainer instance (unused, kept for symmetry with other evaluators).
        cache : Dict[str, Any] | None, optional
            Shared evaluator cache used to reuse prepared explanation context.

        Returns
        -------
        Dict[str, float]
            Mapping with a single deletion-check entry in [0, 1].
        """
        metric_input = MetricInput.from_results(
            model=model,
            explanation_results=explanation_results,
            dataset=dataset,
            explainer=explainer,
            cache=cache,
        )
        return self._evaluate(metric_input)

    def _evaluate(self, metric_input: MetricInput) -> Dict[str, float]:
        """
        Return correctness score using a fully prepared MetricInput.

        Parameters
        ----------
        metric_input : MetricInput
            Unified payload describing the model/dataset/explanations context.

        Returns
        -------
        Dict[str, float]
            Dictionary with the averaged correctness score (or per-instance value
            when ``explanation_idx`` is provided).
        """
        if metric_input.method not in self.supported_methods:
            self.logger.info(
                "CorrectnessEvaluator skipped: method '%s' not a feature-attribution explainer",
                metric_input.method,
            )
            return self._empty_result()

        explanations = metric_input.explanations
        if not explanations:
            return self._empty_result()

        context_cache: Optional[Dict[int, _CorrectnessContext]] = (
            metric_input.cache_bucket("correctness_context") if self.cache_context else None
        )

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return self._empty_result()
            score = self._feature_removal_score(
                metric_input.model,
                explanations[idx],
                context_cache,
            )
            return {self.metric_key: float(score) if score is not None else 0.0}

        scores: List[float] = []
        for explanation in explanations:
            score = self._feature_removal_score(
                metric_input.model,
                explanation,
                context_cache,
            )
            if score is not None:
                scores.append(score)

        correctness = float(np.mean(scores)) if scores else 0.0
        return {self.metric_key: correctness}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _empty_result(self) -> Dict[str, float]:
        return {self.metric_key: 0.0}

    def _feature_removal_score(
        self,
        model: Any,
        explanation: Dict[str, Any],
        context_cache: Optional[Dict[int, _CorrectnessContext]] = None,
    ) -> Optional[float]:
        """Mask the most important features and return the resulting normalized drop."""
        context = self._prepare_context(model, explanation, context_cache)
        if context is None:
            return None

        if self.fast_mode:
            return self._fast_feature_removal_score(
                model,
                explanation,
                context,
            )
        return self._slow_feature_removal_score(model, context)

    def _prepare_context(
        self,
        model: Any,
        explanation: Dict[str, Any],
        context_cache: Optional[Dict[int, _CorrectnessContext]],
    ) -> Optional[_CorrectnessContext]:
        cache_key = id(explanation)
        if context_cache is not None and cache_key in context_cache:
            return context_cache[cache_key]

        importance_vec = self._feature_importance_vector(explanation)
        if importance_vec is None or importance_vec.size == 0:
            return None

        instance = self._extract_instance(explanation)
        if instance is None:
            return None

        k = self._num_features_to_mask(len(importance_vec))
        top_indices = top_k_mask_indices(importance_vec, k)

        baseline = self._baseline_vector(explanation, instance)

        target_class = self._target_class(explanation, model, instance)
        orig_pred = self._prediction_value(
            model,
            explanation,
            instance,
            target_class=target_class,
        )
        if orig_pred is None:
            return None

        context = _CorrectnessContext(
            importance=importance_vec,
            instance=instance,
            baseline=baseline,
            top_indices=top_indices,
            original_prediction=orig_pred,
            target_class=target_class,
        )
        if context_cache is not None:
            context_cache[cache_key] = context
        return context

    def _slow_feature_removal_score(
        self,
        model: Any,
        context: _CorrectnessContext,
    ) -> Optional[float]:
        if context.baseline is None:
            return None

        perturbed = mask_feature_indices(
            context.instance,
            context.top_indices,
            context.baseline,
        )

        try:
            new_pred = self._model_prediction(
                model,
                perturbed,
                target_class=context.target_class,
            )
        except Exception as exc:
            self.logger.debug("CorrectnessEvaluator failed to perturb instance: %s", exc)
            return None

        return self._normalised_change(context.original_prediction, new_pred)

    def _fast_feature_removal_score(
        self,
        model: Any,
        explanation: Dict[str, Any],
        context: _CorrectnessContext,
    ) -> Optional[float]:
        baseline = self._fast_baseline_vector(explanation, context.instance)
        perturbed = mask_feature_indices(
            context.instance,
            context.top_indices,
            baseline,
        )

        orig_pred = self._fast_original_prediction(model, explanation, context)
        if orig_pred is None:
            return None

        try:
            new_pred = self._fast_model_prediction(
                model,
                perturbed,
                target_class=context.target_class,
            )
        except Exception as exc:
            self.logger.debug("CorrectnessEvaluator fast mode failed: %s", exc)
            return None

        return self._normalised_change(orig_pred, new_pred)

    def _normalised_change(self, orig_pred: float, new_pred: float) -> Optional[float]:
        change = abs(orig_pred - new_pred)
        if abs(orig_pred) < 1e-12:
            self.logger.debug(
                "CorrectnessEvaluator denominator nearly zero (orig=%s); skipping instance",
                orig_pred,
            )
            return None
        denom = abs(orig_pred) + 1e-8
        score = float(np.clip(change / denom, 0.0, 1.0))
        if np.isnan(score):
            self.logger.debug(
                "CorrectnessEvaluator produced NaN score (orig=%s, new=%s, denom=%s)",
                orig_pred,
                new_pred,
                denom,
            )
            return None
        return score

    def _fast_original_prediction(
        self,
        model: Any,
        explanation: Dict[str, Any],
        context: _CorrectnessContext,
    ) -> Optional[float]:
        if context.original_prediction is not None:
            return float(context.original_prediction)
        try:
            return self._prediction_value(
                model,
                explanation,
                context.instance,
                target_class=context.target_class,
            )
        except Exception:
            return None

    def _fast_model_prediction(
        self,
        model: Any,
        instance: np.ndarray,
        *,
        target_class: int | None,
    ) -> float:
        return self._model_prediction(model, instance, target_class=target_class)

    def _fast_baseline_vector(
        self,
        explanation: Dict[str, Any],
        instance: np.ndarray,
    ) -> np.ndarray:
        # Fast mode must preserve the same masking semantics as the slow path.
        return self._baseline_vector(explanation, instance)

    def _num_features_to_mask(self, n_features: int) -> int:
        """
        Determine how many top-ranked features to mask based on the evaluator
        configuration (fractional removal or fixed count).
        """
        if self._removal_mode == "count":
            k = max(self.min_features, self._removal_count)
        else:
            frac = self.removal_fraction if self.removal_fraction is not None else 0.0
            k = max(self.min_features, int(np.ceil(frac * n_features)))
        return max(1, min(k, n_features))

    def _feature_importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract feature-importance/attribution vector from a standardized explanation dict.
        Looks in both the root of the explanation and inside metadata so we can support
        multiple explainer schemas.
        """
        arr = extract_attribution_vector(
            explanation,
            logger=self.logger,
            log_prefix="CorrectnessEvaluator",
        )
        if arr is not None:
            return arr
        return None

    def _extract_instance(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        return extract_instance_vector(explanation)

    def _baseline_vector(self, explanation: Dict[str, Any], instance: np.ndarray) -> np.ndarray:
        """Match the original correctness fallback: explainer baseline or constant."""
        return resolve_baseline_vector(
            explanation,
            instance,
            strategy=BASELINE_STRATEGY_EXPLAINER_ONLY,
            default_baseline=self.default_baseline,
            dataset=None,
            logger=self.logger,
            log_prefix="CorrectnessEvaluator",
        )

    def _target_class(
        self,
        explanation: Dict[str, Any],
        model: Any,
        instance: np.ndarray,
    ) -> int | None:
        """
        Resolve the class whose score should be tracked under deletion.

        This uses the shared resolver so ground-truth labels are never reused as
        evaluator targets.
        """
        return resolve_explained_class(explanation, model=model, instance=instance)

    def _prediction_value(
        self,
        model: Any,
        explanation: Dict[str, Any],
        instance: np.ndarray,
        *,
        target_class: int | None,
    ) -> Optional[float]:
        """
        Scalar value to track under feature removal.
        Prefer class probability if available; otherwise use the raw prediction.
        """
        value = resolve_scalar_prediction_score(
            explanation,
            model=model,
            instance=instance,
            target_class=target_class,
            prefer_probability=True,
        )
        if value is not None:
            return float(value)
        self.logger.debug("CorrectnessEvaluator missing prediction value in explanation")
        return None

    def _model_prediction(
        self,
        model: Any,
        instance: np.ndarray,
        *,
        target_class: int | None,
    ) -> float:
        """
        Compute scalar prediction for a perturbed instance, aligned with _prediction_value:
        prefer class probability if available; otherwise use raw prediction.
        """
        return float(
            model_prediction(
                model,
                instance,
                target_class=target_class,
                prefer_probability=True,
            )
        )
