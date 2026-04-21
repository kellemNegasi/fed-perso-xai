"""
Monotonicity correlation metric adapted from Quantus.

This implementation borrows the high-level procedure from Quantus'
``MonotonicityCorrelation`` metric (Nguyen & Rodríguez Martínez,
"On quantitative aspects of model interpretability", 2020). It runs the
same perturbation-by-ranked-attribution routine but plugs into the
``fed_perso_xai`` explainer/evaluator API.
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
from .perturbation import build_metric_rng, chunk_indices
from .prediction_utils import (
    extract_prediction_value,
    model_prediction,
    model_predictions,
)


_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


@dataclass
class _SpearmanResult:
    coefficient: float


@dataclass
class _MonotonicityContext:
    """Cached payload with everything needed to score an explanation."""

    importance: np.ndarray
    instance: np.ndarray
    baseline: np.ndarray
    feature_scale: np.ndarray
    original_prediction: float
    target_class: int | None


class MonotonicityEvaluator(MetricCapabilities):
    """
    Measures whether attribution magnitudes are monotone with the model's sensitivity.

    The evaluator sorts features by attribution magnitude, perturbs them in ranked
    cumulative groups, and correlates the cumulative attribution mass with the
    squared prediction deltas observed after perturbation. Scores in [-1, 1]
    mirror the Quantus ``MonotonicityCorrelation`` metric: +1 when larger
    attributions consistently correspond to larger model impact, -1 for inverted
    rankings. This makes the metric a faithfulness check on feature ordering, not
    just on one-step deletion magnitude.
    """

    per_instance = True
    requires_full_batch = False
    metric_names = ("monotonicity",)
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        nr_samples: int = 1,
        features_in_step: int = 1,
        eps: float = 1e-5,
        default_baseline: float = 0.0,
        abs_attributions: bool = True,
        normalise: bool = True,
        noise_scale: float = 0.0,
        random_state: Optional[int] = None,
        cache_context: bool = True,
    ) -> None:
        if nr_samples < 1:
            raise ValueError("nr_samples must be >= 1.")
        if features_in_step < 1:
            raise ValueError("features_in_step must be >= 1.")
        if eps <= 0:
            raise ValueError("eps must be > 0.")
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be >= 0.")

        self.nr_samples = int(nr_samples)
        self.features_in_step = int(features_in_step)
        self.eps = float(eps)
        self.default_baseline = float(default_baseline)
        self.abs_attributions = bool(abs_attributions)
        self.normalise = bool(normalise)
        self.noise_scale = float(noise_scale)
        self.random_state = int(random_state) if random_state is not None else None
        self.cache_context = bool(cache_context)
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        model: Any,
        explanation_results: Dict[str, Any],
        dataset: Any | None = None,
        explainer: Any | None = None,
        cache: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        """
        Run the monotonicity correlation metric on compatible explanations.

        Parameters
        ----------
        model : Any
            Trained model used to recompute perturbed predictions.
        explanation_results : Dict[str, Any]
            Output dictionary from ``explain_dataset`` (must include ``method`` and
            ``explanations`` entries).
        dataset : Any | None, optional
            Optional dataset reference used for baseline fallback / feature scaling.
        explainer : Any | None, optional
            Unused (interface parity).
        cache : Dict[str, Any] | None, optional
            Shared evaluator cache used to reuse prepared explanation context.
        """
        metric_input = MetricInput.from_results(
            model=model,
            explanation_results=explanation_results,
            dataset=dataset,
            explainer=explainer,
            cache=cache,
        )
        return self._evaluate(metric_input)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _evaluate(self, metric_input: MetricInput) -> Dict[str, float]:
        if metric_input.method not in self.supported_methods:
            self.logger.info(
                "MonotonicityEvaluator skipped: method '%s' not a feature-attribution explainer",
                metric_input.method,
            )
            return {"monotonicity": 0.0}

        explanations = metric_input.explanations
        if not explanations:
            return {"monotonicity": 0.0}

        context_cache: Optional[Dict[int, _MonotonicityContext]] = (
            metric_input.cache_bucket("monotonicity_context") if self.cache_context else None
        )

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return {"monotonicity": 0.0}
            score = self._monotonicity_score(
                metric_input.model,
                explanations[idx],
                explanation_index=idx,
                dataset=metric_input.dataset,
                context_cache=context_cache,
            )
            return {"monotonicity": float(score) if score is not None else 0.0}

        scores: List[float] = []
        for idx, explanation in metric_input.iter_explanations():
            score = self._monotonicity_score(
                metric_input.model,
                explanation,
                explanation_index=idx,
                dataset=metric_input.dataset,
                context_cache=context_cache,
            )
            if score is not None:
                scores.append(score)

        result = float(np.mean(scores)) if scores else 0.0
        return {"monotonicity": result}

    def _monotonicity_score(
        self,
        model: Any,
        explanation: Dict[str, Any],
        *,
        explanation_index: int,
        dataset: Any | None,
        context_cache: Optional[Dict[int, _MonotonicityContext]] = None,
    ) -> Optional[float]:
        # High-level flow:
        # 1. Pull attributions/instance/baseline; skip if malformed.
        # 2. Resolve the scalar prediction target and original output.
        # 3. Sort features by attribution magnitude, then build cumulative subsets
        #    S_t by adding the next most important feature group each step.
        # 4. Correlate cumulative attribution mass with the cumulative squared
        #    prediction deltas, mirroring the paper-aligned monotonicity definition.
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

        # Use a stable descending sort so tied magnitudes preserve feature order
        # deterministically. The correlation definition is about importance ranking,
        # so ordering follows absolute magnitude even when signed attributions are
        # retained for the cumulative attribution signal itself.
        ranked_indices = np.argsort(-np.abs(context.importance), kind="stable")
        ranked_groups = chunk_indices(
            ranked_indices,
            features_per_step=self.features_in_step,
        )
        if len(ranked_groups) < 2:
            return None

        rng = build_metric_rng(self.random_state, offset=explanation_index)

        # Normalize variance terms so low-confidence predictions get amplified while
        # very confident ones contribute less, mirroring Quantus' inverse-probability
        # weighting. Predictions near zero skip the inversion to avoid blow-ups.
        inv_pred = 1.0
        abs_pred = abs(context.original_prediction)
        if abs_pred >= self.eps:
            inv_pred = (1.0 / abs_pred) ** 2

        cumulative_indices: List[np.ndarray] = []
        cumulative_mass = 0.0
        att_sums: List[float] = []
        variance_terms: List[float] = []

        for group in ranked_groups:
            if group.size == 0:
                continue

            cumulative_indices.append(group)
            selected = np.concatenate(cumulative_indices)
            cumulative_mass += float(np.sum(context.importance[group]))

            perturbed_batch = np.repeat(context.instance[np.newaxis, :], self.nr_samples, axis=0)
            replacements = np.repeat(
                context.baseline[selected][np.newaxis, :],
                self.nr_samples,
                axis=0,
            )
            if self.noise_scale > 0.0:
                noise = rng.normal(
                    loc=0.0,
                    scale=self.noise_scale * context.feature_scale[selected],
                    size=(self.nr_samples, selected.size),
                )
                replacements = replacements + noise
            perturbed_batch[:, selected] = replacements

            try:
                preds = model_predictions(
                    model,
                    perturbed_batch,
                    target_class=context.target_class,
                    prefer_probability=True,
                )
            except Exception as exc:
                self.logger.debug(
                    "MonotonicityEvaluator failed to evaluate perturbed batch: %s",
                    exc,
                )
                continue

            preds_arr = np.asarray(preds, dtype=float).reshape(-1)
            preds_arr = preds_arr[np.isfinite(preds_arr)]
            if preds_arr.size == 0:
                continue

            squared_delta = (preds_arr - context.original_prediction) ** 2
            variance = float(np.mean(squared_delta)) * inv_pred
            if not np.isfinite(variance):
                continue

            variance_terms.append(variance)
            att_sums.append(cumulative_mass)

        if len(att_sums) < 2 or len(variance_terms) < 2:
            return None

        att_vec = np.asarray(att_sums, dtype=float)
        var_vec = np.asarray(variance_terms, dtype=float)
        if not np.all(np.isfinite(att_vec)) or not np.all(np.isfinite(var_vec)):
            return None

        spearman = self._spearman(att_vec, var_vec)
        if spearman is None:
            return None
        return float(np.clip(spearman.coefficient, -1.0, 1.0))

    # ------------------------------------------------------------------ #
    # Shared helpers (mirrors other evaluators)                          #
    # ------------------------------------------------------------------ #

    def _prepare_context(
        self,
        model: Any,
        explanation: Dict[str, Any],
        *,
        dataset: Any | None,
        context_cache: Optional[Dict[int, _MonotonicityContext]],
    ) -> Optional[_MonotonicityContext]:
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

        if attrs.size != instance.size:
            self.logger.debug(
                "MonotonicityEvaluator: attribution length %s != instance length %s",
                attrs.size,
                instance.size,
            )
            return None

        baseline = self._baseline_vector(explanation, instance, dataset=dataset)
        target_class = self._target_class(explanation, model, instance)
        original_prediction = self._original_prediction(
            model,
            explanation,
            instance,
            target_class=target_class,
        )
        if original_prediction is None or not np.isfinite(original_prediction):
            return None

        context = _MonotonicityContext(
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
            log_prefix="MonotonicityEvaluator",
        )
        if arr is not None:
            return arr
        self.logger.debug("MonotonicityEvaluator missing importance vector.")
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
            log_prefix="MonotonicityEvaluator",
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

    def _original_prediction(
        self,
        model: Any,
        explanation: Dict[str, Any],
        instance: np.ndarray,
        *,
        target_class: int | None,
    ) -> Optional[float]:
        prediction = extract_prediction_value(
            explanation,
            target_class=target_class,
            prefer_probability=True,
        )
        if prediction is not None:
            return float(prediction)

        try:
            return float(
                model_prediction(
                    model,
                    instance,
                    target_class=target_class,
                    prefer_probability=True,
                )
            )
        except Exception as exc:
            self.logger.debug("MonotonicityEvaluator failed to score instance: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Spearman correlation helper
    # ------------------------------------------------------------------ #

    def _spearman(self, a: np.ndarray, b: np.ndarray) -> Optional[_SpearmanResult]:
        if a.size != b.size or a.size < 2:
            return None

        ranks_a = self._rankdata(a)
        ranks_b = self._rankdata(b)

        diff_a = ranks_a - ranks_a.mean()
        diff_b = ranks_b - ranks_b.mean()
        denom = np.sqrt(np.sum(diff_a**2) * np.sum(diff_b**2))
        if denom == 0.0:
            return None

        rho = float(np.sum(diff_a * diff_b) / denom)
        return _SpearmanResult(coefficient=rho)

    def _rankdata(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(float).ravel()
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(arr.size, dtype=float)
        _, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
        for val_idx, count in enumerate(counts):
            if count <= 1:
                continue
            tie_positions = np.where(inverse == val_idx)[0]
            mean_rank = float(np.mean(ranks[tie_positions]))
            ranks[tie_positions] = mean_rank
        return ranks
