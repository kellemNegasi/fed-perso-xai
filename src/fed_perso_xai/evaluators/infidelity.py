"""
Infidelity metric adapted from Quantus / Yeh et al. (2019).
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
    prepare_attributions,
)
from .base_metric import MetricCapabilities, MetricInput
from .baselines import baseline_vector, feature_scale
from .perturbation import build_metric_rng, sample_random_mask_indices
from .prediction_utils import model_prediction


_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


@dataclass
class _InfidelityContext:
    """Cached payload with everything needed to score an explanation."""

    importance: np.ndarray
    instance: np.ndarray
    baseline: np.ndarray
    feature_scale: np.ndarray
    original_prediction: float
    target_class: int | None


class InfidelityEvaluator(MetricCapabilities):
    """
    Measures explanation infidelity following Yeh et al. (2019).

    For each instance and perturbation sample, we:
      1. Select a subset of features to perturb (replace with a baseline).
      2. Compute the input delta (original minus perturbed) and take the dot
         product with the attribution vector.
      3. Compare that estimate against the actual prediction change induced by
         the perturbation; the squared difference is the infidelity loss.
    Lower scores indicate the attribution vector accurately predicts the model's
    behaviour when the input is perturbed.
    """

    per_instance = True
    requires_full_batch = False
    metric_names = ("infidelity",)
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        n_perturb_samples: int = 16,
        features_per_sample: int = 1,
        default_baseline: float = 0.0,
        abs_attributions: bool = False,
        normalise: bool = False,
        noise_scale: float = 0.0,
        random_state: Optional[int] = None,
        cache_context: bool = True,
    ) -> None:
        if n_perturb_samples < 1:
            raise ValueError("n_perturb_samples must be >= 1.")
        if features_per_sample < 1:
            raise ValueError("features_per_sample must be >= 1.")
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be >= 0.")

        self.n_perturb_samples = int(n_perturb_samples)
        self.features_per_sample = int(features_per_sample)
        self.default_baseline = float(default_baseline)
        self.abs_attributions = bool(abs_attributions)
        self.normalise = bool(normalise)
        self.noise_scale = float(noise_scale)
        self.random_state = int(random_state) if random_state is not None else None
        self.cache_context = bool(cache_context)
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

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
            self.logger.info(
                "InfidelityEvaluator skipped: method '%s' not in feature-attribution set",
                metric_input.method,
            )
            return {"infidelity": 0.0}

        explanations = metric_input.explanations
        if not explanations:
            return {"infidelity": 0.0}

        context_cache: Optional[Dict[int, _InfidelityContext]] = (
            metric_input.cache_bucket("infidelity_context") if self.cache_context else None
        )

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return {"infidelity": 0.0}
            score = self._infidelity_score(
                metric_input.model,
                explanations[idx],
                explanation_index=idx,
                dataset=metric_input.dataset,
                context_cache=context_cache,
            )
            return {"infidelity": float(score) if score is not None else 0.0}

        scores: List[float] = []
        for idx, explanation in metric_input.iter_explanations():
            score = self._infidelity_score(
                metric_input.model,
                explanation,
                explanation_index=idx,
                dataset=metric_input.dataset,
                context_cache=context_cache,
            )
            if score is not None:
                scores.append(score)

        return {"infidelity": float(np.mean(scores)) if scores else 0.0}

    # ------------------------------------------------------------------ #
    # Core metric                                                        #
    # ------------------------------------------------------------------ #

    def _infidelity_score(
        self,
        model: Any,
        explanation: Dict[str, Any],
        *,
        explanation_index: int,
        dataset: Any | None = None,
        context_cache: Optional[Dict[int, _InfidelityContext]] = None,
    ) -> Optional[float]:
        context = self._prepare_context(
            model,
            explanation,
            dataset=dataset,
            context_cache=context_cache,
        )
        if context is None:
            return None

        n_features = context.instance.size
        if n_features == 0:
            return None

        mask_size = min(self.features_per_sample, n_features)
        rng = build_metric_rng(self.random_state, offset=explanation_index)
        errors: List[float] = []

        for _ in range(self.n_perturb_samples):
            chosen = self._sample_feature_indices(rng, n_features=n_features, mask_size=mask_size)
            perturbed = context.instance.copy()
            replacement = context.baseline[chosen].copy()
            if self.noise_scale > 0.0:
                noise = rng.normal(
                    loc=0.0,
                    scale=self.noise_scale * context.feature_scale[chosen],
                    size=chosen.size,
                )
                replacement = replacement + noise
            perturbed[chosen] = replacement

            delta = context.instance - perturbed
            approx_change = float(np.dot(context.importance, delta))
            if not np.isfinite(approx_change):
                continue

            try:
                new_pred = self._model_prediction(
                    model,
                    perturbed,
                    target_class=context.target_class,
                )
            except Exception as exc:
                self.logger.debug(
                    "InfidelityEvaluator failed to evaluate perturbed instance: %s",
                    exc,
                )
                continue

            true_change = float(context.original_prediction - new_pred)
            if not np.isfinite(true_change):
                continue

            error = float((approx_change - true_change) ** 2)
            if np.isfinite(error):
                errors.append(error)

        if not errors:
            return None
        return float(np.mean(errors))

    # ------------------------------------------------------------------ #
    # Shared helpers (mirrors other evaluators)                          #
    # ------------------------------------------------------------------ #

    def _prepare_context(
        self,
        model: Any,
        explanation: Dict[str, Any],
        *,
        dataset: Any | None,
        context_cache: Optional[Dict[int, _InfidelityContext]],
    ) -> Optional[_InfidelityContext]:
        cache_key = id(explanation)
        if context_cache is not None and cache_key in context_cache:
            return context_cache[cache_key]

        attrs = self._feature_importance_vector(explanation)
        if attrs is None or attrs.size == 0:
            return None
        attrs = prepare_attributions(
            attrs,
            abs_attributions=self.abs_attributions,
            normalise=self.normalise,
        )

        instance = self._extract_instance(explanation)
        if instance is None:
            return None

        if instance.size != attrs.size:
            self.logger.debug(
                "InfidelityEvaluator length mismatch: instance=%s, attrs=%s",
                instance.size,
                attrs.size,
            )
            return None

        baseline = self._baseline_vector(explanation, instance, dataset=dataset)
        target_class = self._target_class(explanation, model, instance)

        try:
            original_prediction = self._model_prediction(
                model,
                instance,
                target_class=target_class,
            )
        except Exception as exc:
            self.logger.debug("InfidelityEvaluator failed to score instance: %s", exc)
            return None

        context = _InfidelityContext(
            importance=attrs,
            instance=instance,
            baseline=baseline,
            feature_scale=feature_scale(instance),
            original_prediction=float(original_prediction),
            target_class=target_class,
        )
        if context_cache is not None:
            context_cache[cache_key] = context
        return context

    def _feature_importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        arr = extract_attribution_vector(
            explanation,
            logger=self.logger,
            log_prefix="InfidelityEvaluator",
        )
        if arr is not None:
            return arr
        self.logger.debug("InfidelityEvaluator missing attribution vector.")
        return None

    def _extract_instance(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        return extract_instance_vector(explanation)

    def _baseline_vector(
        self,
        explanation: Dict[str, Any],
        instance: np.ndarray,
        *,
        dataset: Any | None,
    ) -> np.ndarray:
        return baseline_vector(
            explanation,
            instance,
            default_baseline=self.default_baseline,
            dataset=dataset,
            logger=self.logger,
            log_prefix="InfidelityEvaluator",
        )

    def _sample_feature_indices(
        self,
        rng: np.random.Generator,
        *,
        n_features: int,
        mask_size: int,
    ) -> np.ndarray:
        if mask_size >= n_features:
            return np.arange(n_features, dtype=int)
        return sample_random_mask_indices(
            rng,
            n_features=n_features,
            mask_size=mask_size,
        )

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
