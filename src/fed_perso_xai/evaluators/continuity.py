"""
Continuity metric – stability for slight variations.

Adapts the non-sensitivity test: apply a small perturbation to an instance,
recompute the explanation for the perturbed sample, and measure similarity with
the original attribution vector.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .attribution_utils import (
    FEATURE_METHOD_KEYS,
    extract_attribution_vector,
    extract_instance_vector,
)
from .base_metric import MetricCapabilities, MetricInput
from .baselines import dataset_feature_std
from .perturbation import (
    add_scaled_gaussian_noise,
    approximate_perturbed_attributions,
    build_metric_rng,
)


_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


class ContinuityEvaluator(MetricCapabilities):
    """Continuity estimator that perturbs inputs and checks attribution stability."""

    per_instance = True
    requires_full_batch = False
    metric_names = ("continuity_stability",)
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        max_instances: int = 5,
        noise_scale: float = 0.01,
        metric_key: str = "continuity_stability",
        random_state: Optional[int] = 42,
    ) -> None:
        """
        Parameters
        ----------
        max_instances : int, optional
            Maximum number of explanations to perturb per call (defaults to 5).
        noise_scale : float, optional
            Standard-deviation multiplier for the Gaussian noise added to each
            instance; larger values stress continuity more aggressively.
        metric_key : str, optional
            Name of the metric emitted in the returned dictionary.
        random_state : int | None, optional
            Seed for the random number generator (None for stochastic runs).
        """
        self.max_instances = max(1, int(max_instances))
        self.noise_scale = float(max(0.0, noise_scale))
        self.metric_key = metric_key or "continuity_stability"
        self.random_state = random_state
        self.metric_names = (self.metric_key,)
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
        Perturb instances and measure how similar their explanations remain.

        Parameters
        ----------
        model : Any
            Present for API symmetry; unused by the current metric.
        explanation_results : Dict[str, Any]
            Output from ``BaseExplainer.explain_dataset`` (may include ``current_index``).
        dataset : Any | None, optional
            Dataset reference used to estimate feature-wise standard deviations.
        explainer : Any | None, optional
            Explainer instance used to re-run explanations on perturbed inputs.
        cache : Dict[str, Any] | None, optional
            Shared evaluator cache. Currently unused directly, but accepted for
            compatibility with the shared metric execution path.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the continuity score keyed by ``metric_key``.
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
        Compute the continuity score given a MetricInput payload.

        Parameters
        ----------
        metric_input : MetricInput
            Standardized evaluator context containing explanations and metadata.

        Returns
        -------
        Dict[str, float]
            Dictionary with the continuity score (or zero if inputs invalid).
        """
        if metric_input.method not in self.supported_methods:
            return self._empty_result()

        explanations = metric_input.explanations
        if not explanations:
            return self._empty_result()

        feature_std = dataset_feature_std(metric_input.dataset, explanations[0])

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return self._empty_result()
            rng = build_metric_rng(self.random_state, offset=idx)
            score = self._continuity_score(
                explanation=explanations[idx],
                feature_std=feature_std,
                rng=rng,
                explainer=metric_input.explainer,
            )
            return {self.metric_key: float(score) if score is not None else 0.0}

        rng = build_metric_rng(self.random_state)
        similarities: List[float] = []
        n_samples = min(self.max_instances, len(explanations))

        for i in range(n_samples):
            score = self._continuity_score(
                explanation=explanations[i],
                feature_std=feature_std,
                rng=rng,
                explainer=metric_input.explainer,
            )
            if score is not None:
                similarities.append(score)

        score = float(np.mean(similarities)) if similarities else 0.0
        return {self.metric_key: score}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _empty_result(self) -> Dict[str, float]:
        return {self.metric_key: 0.0}

    def _instance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract the instance represented by an explanation as a 1-D numpy vector."""
        return extract_instance_vector(explanation)

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Return the attribution vector from either the explanation or metadata."""
        return extract_attribution_vector(
            explanation,
            logger=self.logger,
            log_prefix="ContinuityEvaluator",
        )

    def _continuity_score(
        self,
        explanation: Dict[str, Any],
        feature_std: Optional[np.ndarray],
        rng: np.random.Generator,
        explainer: Any | None,
    ) -> Optional[float]:
        """Return absolute correlation between original and perturbed explanations."""
        instance = self._instance_vector(explanation)
        importance = self._importance_vector(explanation)
        if instance is None or importance is None or importance.size == 0:
            return None

        perturbed_instance = add_scaled_gaussian_noise(
            instance,
            feature_std=feature_std,
            noise_scale=self.noise_scale,
            rng=rng,
        )

        perturbed_importance = self._true_perturbed_importance(
            explainer=explainer,
            perturbed_instance=perturbed_instance,
        )
        if perturbed_importance is None or perturbed_importance.size != importance.size:
            return None

        corr = self._similarity(importance, perturbed_importance)
        if corr is None or np.isnan(corr):
            return None
        return abs(corr)

    def _true_perturbed_importance(
        self,
        explainer: Any | None,
        perturbed_instance: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Re-run the explainer on the perturbed instance (if available) to obtain the
        actual attribution vector used in the continuity comparison.
        """
        if explainer is None or not hasattr(explainer, "explain_instance"):
            return None

        try:
            perturbed_expl = explainer.explain_instance(perturbed_instance)
        except Exception as exc:
            self.logger.debug(
                "ContinuityEvaluator failed to re-explain perturbed instance: %s", exc
            )
            return None
        return self._importance_vector(perturbed_expl)

    def _approximate_perturbed_importance(
        self,
        original_instance: np.ndarray,
        perturbed_instance: np.ndarray,
        original_importance: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Scale the original importance vector according to relative input changes."""
        return approximate_perturbed_attributions(
            original_instance=original_instance,
            perturbed_instance=perturbed_instance,
            original_importance=original_importance,
        )

    def _similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
        """Compute Pearson correlation between two attribution vectors if defined."""
        a = np.asarray(vec_a, dtype=float).reshape(-1)
        b = np.asarray(vec_b, dtype=float).reshape(-1)
        if a.size < 2 or b.size < 2 or a.size != b.size:
            return None
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return None
        std_a = float(np.std(a))
        std_b = float(np.std(b))
        if std_a < 1e-8 or std_b < 1e-8:  # prevent division by zero, np.corrcoef would complain.
            return None
        try:
            corr = np.corrcoef(a, b)[0, 1]
            return float(corr)
        except Exception:
            return None
