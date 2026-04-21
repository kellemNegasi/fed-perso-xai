# src/evaluators/compactness.py
"""
Compactness metrics for local feature-attribution explanations.

Focuses on the Size family from Co-12 Section 6.7:
    • Sparsity (fraction of near-zero attributions)
    • Top-k coverage (importance mass captured by top 5 / top 10 features)
    • Effective feature count (inverse participation ratio, normalized)
Adapted from:  https://github.com/umberH/benchmarking  CompactnessEvaluator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from .attribution_utils import FEATURE_METHOD_KEYS, extract_attribution_vector
from .base_metric import MetricCapabilities, MetricInput

_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


class CompactnessEvaluator(MetricCapabilities):
    """
    Aggregates Size-style compactness indicators for feature attributions.
    Parameters
    ----------
    zero_tolerance : float, optional
        Magnitude threshold that defines a "near-zero" attribution.
    """

    per_instance = True
    requires_full_batch = False
    metric_names = (
        "compactness_sparsity",
        "compactness_top5_coverage",
        "compactness_top10_coverage",
        "compactness_effective_features",
    )
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(self, *, zero_tolerance: float = 1e-8) -> None:
        """Store the magnitude threshold that defines a "near-zero" attribution."""
        self.zero_tolerance = max(0.0, float(zero_tolerance))
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
        Return compactness metrics for a batch (or single) explanation.

        Parameters
        ----------
        model : Any
            Trained model tied to the explanations (unused but kept for parity).
        explanation_results : Dict[str, Any]
            Result dictionary produced by ``explain_dataset``.
        dataset : Any | None, optional
            Dataset reference (unused placeholder).
        explainer : Any | None, optional
            Explainer instance (unused placeholder).
        cache : Dict[str, Any] | None, optional
            Shared evaluator cache (unused placeholder, kept for framework parity).

        Returns
        -------
        Dict[str, float]
            Mapping for each compactness metric key defined in ``metric_names``.
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
        Compute compactness metrics using a MetricInput payload.

        Parameters
        ----------
        metric_input : MetricInput
            Standardized evaluator input containing explanations/context.

        Returns
        -------
        Dict[str, float]
            Averaged compactness scores (or per-instance values when an index is set).
        """
        if metric_input.method not in self.supported_methods:
            return self._empty_result()

        explanations = metric_input.explanations
        if not explanations:
            return self._empty_result()

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return self._empty_result()
            return self._metrics_for_explanation(explanations[idx])

        accumulators = {key: [] for key in self.metric_names}
        for explanation in explanations:
            metrics = self._metrics_for_explanation(explanation)
            if not metrics:
                continue
            for key, value in metrics.items():
                accumulators[key].append(value)

        return {
            key: float(np.mean(values)) if values else 0.0
            for key, values in accumulators.items()
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _empty_result(self) -> Dict[str, float]:
        """Convenience helper that returns zeros for all compactness metrics."""
        return {key: 0.0 for key in self.metric_names}

    def _metrics_for_explanation(self, explanation: Dict[str, Any]) -> Dict[str, float]:
        """Return compactness metrics for a single explanation."""
        importance = self._importance_vector(explanation)
        if importance is None:
            return {}
        metrics = self._feature_compactness(importance)
        return metrics or {}

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract the attribution vector from an explanation dict or metadata."""
        arr = extract_attribution_vector(explanation)
        if arr is not None:
            return arr
        self.logger.debug("CompactnessEvaluator missing importance vector in explanation")
        return None

    def _feature_compactness(self, importance: np.ndarray) -> Optional[Dict[str, float]]:
        """Compute per-instance sparsity, coverage, and effective-feature scores."""
        if importance.size == 0:
            return None

        imp = np.abs(importance)
        if np.any(np.isnan(imp)) or np.any(np.isinf(imp)):
            return None

        n_features = imp.size
        total = float(np.sum(imp))
        tol = self.zero_tolerance

        non_zero = np.count_nonzero(imp > tol)
        # Sparsity: fraction of features whose magnitude is below the tolerance.
        sparsity = 1.0 - (non_zero / n_features)
        sparsity = float(np.clip(sparsity, 0.0, 1.0))

        if total <= 0.0:
            top5 = 0.0
            top10 = 0.0
            effective = 0.0
        else:
            sorted_imp = np.sort(imp)[::-1]
            normalized = sorted_imp / total
            # Top-k coverage: proportion of attribution mass captured by the k largest
            # features, indicating how concentrated the explanation is.
            top5 = float(np.sum(normalized[: min(5, n_features)]))
            top10 = float(np.sum(normalized[: min(10, n_features)]))

            prob_dist = imp / total
            participation = float(np.sum(prob_dist ** 2))  # Inverse Participation Ratio,
            # Measure effectivens as the inverse of aprticipation. The hihher the participation, the lower the effective features.
            effective_count = 1.0 / participation if participation > 0.0 else float(n_features)

            if n_features == 1:
                effective = 1.0
            else:
                # Normalize IPO so 1.0 means "one dominant feature" and 0.0 indicates
                # a uniform spread across all features.
                effective = 1.0 - (effective_count - 1.0) / (n_features - 1.0)
                effective = float(np.clip(effective, 0.0, 1.0))

        return {
            "compactness_sparsity": sparsity,
            "compactness_top5_coverage": top5,
            "compactness_top10_coverage": top10,
            "compactness_effective_features": effective,
        }
