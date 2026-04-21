"""
Covariate complexity / regularity metric inspired by Co-12 (Sec. 6.6).

Computes the Shannon entropy of each explanation's attribution vector to quantify
how noisy the feature-importance distribution is. Lower entropy (higher
regularity) indicates concentrated, easier-to-memorize explanations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .attribution_utils import FEATURE_METHOD_KEYS, extract_attribution_vector
from .base_metric import MetricCapabilities, MetricInput

_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


class CovariateComplexityEvaluator(MetricCapabilities):
    """
    Evaluate average Shannon entropy (and its complement) over local explanations.

    Parameters
    ----------
    metric_key : str, optional
        Dictionary key used for the normalized entropy output.
    regularity_key : str, optional
        Dictionary key for the 1 - entropy complement (higher is better).
    """

    def __init__(
        self,
        *,
        metric_key: str = "covariate_complexity",
        regularity_key: str = "covariate_regularity",
    ) -> None:
        """
        Parameters
        ----------
        metric_key : str, optional
            Dictionary key used for the normalized entropy output.
        regularity_key : str, optional
            Dictionary key for the 1 - entropy complement (higher is better).
        """
        self.metric_key = metric_key or "covariate_complexity"
        self.regularity_key = regularity_key or f"{self.metric_key}_regularity"
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
        Compute the mean normalized entropy of feature-importance vectors.

        Parameters
        ----------
        model : Any
            Present for API symmetry; unused by this metric.
        explanation_results : Dict[str, Any]
            Output dict from ``BaseExplainer.explain_dataset`` containing per-instance
            explanations with attribution scores.
        dataset : Any | None, optional
            Dataset reference (unused placeholder).
        explainer : Any | None, optional
            Explainer instance (unused placeholder).
        cache : Dict[str, Any] | None, optional
            Shared evaluator cache (unused placeholder, kept for framework parity).

        Returns
        -------
        Dict[str, float]
            Dictionary containing the averaged complexity and regularity scores.
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
            Aggregated complexity/regularity metrics or zeros if inputs are invalid.
        """
        if metric_input.method not in self.supported_methods:
            return {
                self.metric_key: 0.0,
                self.regularity_key: 0.0,
            }

        explanations = metric_input.explanations
        if not explanations:
            return {
                self.metric_key: 0.0,
                self.regularity_key: 0.0,
            }

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return {
                    self.metric_key: 0.0,
                    self.regularity_key: 0.0,
                }
            metrics = self._metrics_for_explanation(explanations[idx])
            return metrics or {
                self.metric_key: 0.0,
                self.regularity_key: 0.0,
            }

        entropies: List[float] = []
        regularities: List[float] = []

        for explanation in explanations:
            metrics = self._metrics_for_explanation(explanation)
            if not metrics:
                continue
            entropies.append(metrics[self.metric_key])
            regularities.append(metrics[self.regularity_key])

        complexity = float(np.mean(entropies)) if entropies else 0.0
        regularity = float(np.mean(regularities)) if regularities else 0.0
        return {
            self.metric_key: complexity,
            self.regularity_key: regularity,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _metrics_for_explanation(self, explanation: Dict[str, Any]) -> Optional[Dict[str, float]]:
        importance = self._importance_vector(explanation)
        if importance is None or importance.size == 0:
            return None
        if not np.all(np.isfinite(importance)):
            return None

        importance = np.abs(importance)
        total = float(np.sum(importance))
        if total <= 0.0:
            return {
                self.metric_key: 0.0,
                self.regularity_key: 1.0,
            }

        prob = importance / total
        entropy = self._shannon_entropy(prob)
        max_entropy = np.log2(prob.size) if prob.size > 1 else 0.0
        normalized = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        normalized = float(np.clip(normalized, 0.0, 1.0))
        return {
            self.metric_key: normalized,
            self.regularity_key: 1.0 - normalized,
        }

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract feature-importance values regardless of explainer schema by checking
        both the explanation root and its metadata.
        """
        arr = extract_attribution_vector(
            explanation,
            logger=self.logger,
            log_prefix="CovariateComplexityEvaluator",
        )
        if arr is not None:
            return arr
        return None

    def _shannon_entropy(self, prob: np.ndarray) -> float:
        """Return Shannon entropy in bits for a probability vector."""
        safe_prob = np.clip(prob, 1e-12, 1.0)
        entropy = -float(np.sum(safe_prob * np.log2(safe_prob)))
        return entropy

    per_instance = True
    requires_full_batch = False
    metric_names = ("covariate_complexity", "covariate_regularity")
    supported_methods = tuple(_FEATURE_METHOD_KEYS)
