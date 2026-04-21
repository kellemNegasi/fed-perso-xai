"""
Completeness evaluator implementing the deletion-check metric.

Given a feature-attribution explanation, we remove (mask) every feature that the
explanation marks as important and measure how much the model prediction drops.
The score is contrasted against randomly masked feature sets of the same size,
as suggested by the deletion check in the Co-12 survey.
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
from .baselines import BASELINE_STRATEGY_EXPLAINER_ONLY, resolve_baseline_vector
from .perturbation import build_metric_rng, generate_random_masked_batch, support_indices
from .prediction_utils import (
    model_prediction,
    model_predictions,
    resolve_scalar_prediction_score,
)


_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


@dataclass
class _CompletenessContext:
    """Cached payload with everything needed to score an explanation."""

    importance: np.ndarray
    instance: np.ndarray
    baseline: np.ndarray
    mask_indices: np.ndarray
    original_prediction: float
    target_class: int | None


class CompletenessEvaluator(MetricCapabilities):
    """
    Measure completeness by masking the entire attribution support and comparing
    the resulting prediction drop with random deletions of equal size.

    Parameters
    ----------
    magnitude_threshold : float, optional
        Absolute attribution magnitude threshold defining the support.
    min_features : int, optional
        Minimum number of features to mask regardless of threshold.
    random_trials : int, optional
        Number of random deletion baselines to compute per explanation.
    default_baseline : float, optional
        Value used to fill masked features when no explainer baseline is given.
    random_state : int | None, optional
        Seed for the random baseline sampler.
    """

    per_instance = True
    requires_full_batch = False
    metric_names = (
        "completeness_drop",
        "completeness_random_drop",
        "completeness_score",
    )
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        magnitude_threshold: float = 1e-8,
        min_features: int = 1,
        random_trials: int = 5,
        default_baseline: float = 0.0,
        fast_mode: bool = True,
        random_state: Optional[int] = None,
        cache_context: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        magnitude_threshold : float, optional
            Absolute attribution magnitude that defines whether a feature belongs
            to the explanation support. If not enough features pass the threshold,
            the highest-magnitude features are used until ``min_features`` is met.
        min_features : int, optional
            Minimum number of features to mask regardless of threshold filtering.
        random_trials : int, optional
            How many random deletion baselines to compute per explanation.
        default_baseline : float, optional
            Fallback value for masked features when the explanation metadata does
            not provide a per-feature ``baseline_instance``.
        random_state : int | None, optional
            Seed for the random baseline sampler.
        """
        self.magnitude_threshold = float(max(0.0, magnitude_threshold))
        self.min_features = max(1, int(min_features))
        self.random_trials = max(0, int(random_trials))
        self.default_baseline = float(default_baseline)
        self.fast_mode = bool(fast_mode)
        self.random_state = int(random_state) if random_state is not None else None
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
        Compute deletion-check completeness metrics for a batch of explanations.

        Parameters
        ----------
        model : Any
            Trained model that produced the explanations.
        explanation_results : Dict[str, Any]
            Output dict from ``BaseExplainer.explain_dataset``.
        dataset : Any | None, optional
            Dataset reference (unused placeholder).
        explainer : Any | None, optional
            Explainer instance (unused placeholder).
        cache : Dict[str, Any] | None, optional
            Shared evaluator cache used to reuse prepared explanation context.

        Returns
        -------
        Dict[str, float]
            Averaged completeness metrics (drop, random drop, score).
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
        Internal helper operating directly on MetricInput.

        Parameters
        ----------
        metric_input : MetricInput
            Standardized evaluator payload.

        Returns
        -------
        Dict[str, float]
            Aggregated completeness metrics or zeros if inputs are invalid.
        """
        if metric_input.method not in self.supported_methods:
            return self._empty_result()

        explanations = metric_input.explanations
        if not explanations:
            return self._empty_result()

        context_cache: Optional[Dict[int, _CompletenessContext]] = (
            metric_input.cache_bucket("completeness_context") if self.cache_context else None
        )

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return self._empty_result()
            metrics = self._metrics_for_explanation(
                metric_input.model,
                explanations[idx],
                explanation_index=idx,
                context_cache=context_cache,
            )
            return metrics or self._empty_result()

        drops: List[float] = []
        baseline_drops: List[float] = []
        scores: List[float] = []

        for idx, explanation in metric_input.iter_explanations():
            metrics = self._metrics_for_explanation(
                metric_input.model,
                explanation,
                explanation_index=idx,
                context_cache=context_cache,
            )
            if not metrics:
                continue
            drops.append(metrics["completeness_drop"])
            baseline_drops.append(metrics["completeness_random_drop"])
            scores.append(metrics["completeness_score"])

        if not drops:
            return self._empty_result()

        return {
            "completeness_drop": float(np.mean(drops)),
            "completeness_random_drop": float(np.mean(baseline_drops)) if baseline_drops else 0.0,
            "completeness_score": float(np.mean(scores)),
        }

    # ------------------------------------------------------------------ #
    # Helper utilities                                                   #
    # ------------------------------------------------------------------ #

    def _empty_result(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.metric_names}

    def _metrics_for_explanation(
        self,
        model: Any,
        explanation: Dict[str, Any],
        *,
        explanation_index: int,
        context_cache: Optional[Dict[int, _CompletenessContext]] = None,
    ) -> Optional[Dict[str, float]]:
        if self.fast_mode:
            metrics = self._fast_metrics_for_explanation(model, explanation)
            if metrics is not None:
                return metrics
            # fall back to exact computation if heuristic failed

        context = self._prepare_context(model, explanation, context_cache)
        if context is None:
            return None

        target_drop = self._normalized_drop(
            model,
            context.instance,
            context.baseline,
            context.mask_indices,
            context.original_prediction,
            target_class=context.target_class,
        )
        if target_drop is None:
            return None

        random_values = self._random_baseline_drops(
            model,
            context,
            explanation_index=explanation_index,
        )
        random_mean = float(np.mean(random_values)) if random_values else 0.0
        score = max(0.0, target_drop - random_mean)
        return {
            "completeness_drop": target_drop,
            "completeness_random_drop": random_mean,
            "completeness_score": score,
        }

    def _prepare_context(
        self,
        model: Any,
        explanation: Dict[str, Any],
        context_cache: Optional[Dict[int, _CompletenessContext]],
    ) -> Optional[_CompletenessContext]:
        cache_key = id(explanation)
        if context_cache is not None and cache_key in context_cache:
            return context_cache[cache_key]

        importance = self._importance_vector(explanation)
        if importance is None or importance.size == 0:
            return None

        instance = self._extract_instance(explanation)
        if instance is None:
            return None

        baseline = self._baseline_vector(explanation, instance)
        target_class = self._target_class(explanation, model, instance)
        orig_pred = self._prediction_value(
            explanation,
            model=model,
            instance=instance,
            target_class=target_class,
        )
        if orig_pred is None:
            return None

        mask_indices = support_indices(
            importance,
            magnitude_threshold=self.magnitude_threshold,
            min_features=self.min_features,
        )
        if mask_indices.size == 0:
            return None

        context = _CompletenessContext(
            importance=importance,
            instance=instance,
            baseline=baseline,
            mask_indices=mask_indices,
            original_prediction=float(orig_pred),
            target_class=target_class,
        )
        if context_cache is not None:
            context_cache[cache_key] = context
        return context

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        arr = extract_attribution_vector(
            explanation,
            logger=self.logger,
            log_prefix="CompletenessEvaluator",
        )
        if arr is not None:
            return arr
        self.logger.debug("CompletenessEvaluator missing attribution vector.")
        return None

    def _extract_instance(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        return extract_instance_vector(explanation)

    def _baseline_vector(self, explanation: Dict[str, Any], instance: np.ndarray) -> np.ndarray:
        # The original completeness evaluator only consumed an explainer-provided
        # baseline_instance or the configured scalar fallback; keep that behavior
        # rather than deriving a dataset-mean baseline here.
        return resolve_baseline_vector(
            explanation,
            instance,
            strategy=BASELINE_STRATEGY_EXPLAINER_ONLY,
            default_baseline=self.default_baseline,
            dataset=None,
            logger=self.logger,
            log_prefix="CompletenessEvaluator",
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

    def _normalized_drop(
        self,
        model: Any,
        instance: np.ndarray,
        baseline: np.ndarray,
        indices: np.ndarray,
        original_pred: float,
        *,
        target_class: int | None,
    ) -> Optional[float]:
        if indices.size == 0:
            return None
        perturbed = np.asarray(instance, dtype=float).reshape(-1).copy()
        perturbed[np.asarray(indices, dtype=int)] = baseline[np.asarray(indices, dtype=int)]
        try:
            new_pred = self._model_prediction(
                model,
                perturbed,
                target_class=target_class,
            )
        except Exception as exc:
            self.logger.debug("CompletenessEvaluator failed to evaluate perturbed instance: %s", exc)
            return None

        if abs(original_pred) < 1e-12:
            self.logger.debug(
                "CompletenessEvaluator denominator nearly zero (orig=%s); skipping instance",
                original_pred,
            )
            return None
        denom = abs(original_pred) + 1e-8
        drop = abs(original_pred - new_pred) / denom
        if np.isnan(drop) or np.isinf(drop):
            return None
        return float(np.clip(drop, 0.0, 1.0))

    def _random_baseline_drops(
        self,
        model: Any,
        context: _CompletenessContext,
        *,
        explanation_index: int,
    ) -> List[float]:
        mask_size = int(context.mask_indices.size)
        n_features = int(context.importance.size)
        if self.random_trials <= 0 or mask_size <= 0 or mask_size > n_features:
            return []

        rng = build_metric_rng(self.random_state, offset=explanation_index)
        batch = generate_random_masked_batch(
            context.instance,
            context.baseline,
            n_trials=self.random_trials,
            mask_size=mask_size,
            rng=rng,
        )

        try:
            preds = self._model_predictions(
                model,
                batch,
                target_class=context.target_class,
            )
        except Exception as exc:
            self.logger.debug("CompletenessEvaluator random baseline failed: %s", exc)
            return []

        if abs(context.original_prediction) < 1e-12:
            return []
        denom = abs(context.original_prediction) + 1e-8
        drops = np.abs(context.original_prediction - preds) / denom
        drops = drops[np.isfinite(drops)]
        valid = np.clip(drops, 0.0, 1.0)
        return valid.tolist()

    def _model_predictions(
        self,
        model: Any,
        instances: np.ndarray,
        *,
        target_class: int | None,
    ) -> np.ndarray:
        return np.asarray(
            model_predictions(
                model,
                instances,
                target_class=target_class,
                prefer_probability=True,
            ),
            dtype=float,
        )

    def _fast_metrics_for_explanation(
        self,
        model: Any,
        explanation: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        importance = self._importance_vector(explanation)
        if importance is None or importance.size == 0:
            return None

        metadata = explanation.get("metadata") or {}
        text_content = explanation.get("text_content") or metadata.get("text_content")

        if text_content:
            score = self._text_completeness_score(text_content, importance)
            return {
                "completeness_drop": score,
                "completeness_random_drop": 0.0,
                "completeness_score": score,
            }

        instance = self._extract_instance(explanation)
        target_class = resolve_explained_class(explanation, model=model, instance=instance)
        prediction_value = self._prediction_value(
            explanation,
            model=model,
            instance=instance,
            target_class=target_class,
        )
        if prediction_value is None:
            return None

        baseline_prediction = None
        baseline_provided = False
        if "baseline_prediction" in metadata:
            baseline_prediction = metadata.get("baseline_prediction")
            baseline_provided = True
        elif "expected_value" in metadata:
            baseline_prediction = metadata.get("expected_value")
            baseline_provided = True
        if isinstance(baseline_prediction, (list, tuple, np.ndarray)):
            baseline_arr = np.asarray(baseline_prediction).ravel()
            if baseline_arr.size == 0:
                baseline_prediction = 0.0
            elif target_class is not None and 0 <= int(target_class) < baseline_arr.size:
                baseline_prediction = float(baseline_arr[int(target_class)])
            else:
                baseline_prediction = float(baseline_arr[0])
        if baseline_prediction is None and not baseline_provided:
            return None
        baseline_prediction = float(baseline_prediction or 0.0)

        score = self._tabular_completeness_score(
            importance, prediction_value, float(baseline_prediction)
        )
        return {
            "completeness_drop": score,
            "completeness_random_drop": 0.0,
            "completeness_score": score,
        }

    def _text_completeness_score(self, text: str, importance: np.ndarray) -> float:
        words = text.split()
        if not words:
            return 0.0
        vec = np.abs(importance[: len(words)])
        if vec.size == 0:
            return 0.0

        percentile_threshold = np.percentile(vec, 80)
        important_words = np.count_nonzero(vec >= percentile_threshold)
        coverage = important_words / len(words)

        total = float(np.sum(vec))
        if total > 1e-12:
            normalized = vec / total
            entropy = -np.sum(normalized * np.log(normalized + 1e-10))
            max_entropy = np.log(len(words)) if len(words) > 1 else 0.0
            entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0
            simpson = np.sum(normalized**2)
            effective_words = 1.0 / simpson if simpson > 0 else len(words)
            effective_score = effective_words / len(words)
        else:
            entropy_score = 0.0
            effective_score = 0.0

        score = np.mean([coverage, entropy_score, effective_score])
        return float(np.clip(score, 0.0, 1.0))

    def _tabular_completeness_score(
        self,
        importance: np.ndarray,
        prediction_value: float,
        baseline_prediction: float,
    ) -> float:
        sum_attributions = float(np.sum(importance))
        output_diff = float(prediction_value - baseline_prediction)
        if abs(output_diff) > 1e-8:
            score = 1.0 - abs(sum_attributions - output_diff) / abs(output_diff)
        else:
            score = 1.0 if abs(sum_attributions) < 1e-8 else 0.0
        return float(np.clip(score, 0.0, 1.0))

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
