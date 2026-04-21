"""
Contrastivity metric for local feature-attribution explanations on tabular data.

Measures how dissimilar attribution patterns are across predictions for different
classes using a Structural Similarity Index Measure (SSIM) variant. Inspired by
the Random Logit / target-sensitivity checks described by Sixt et al. (2020).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .attribution_utils import (
    FEATURE_METHOD_KEYS,
    extract_attribution_vector,
    prepare_attributions,
)
from .base_metric import MetricCapabilities, MetricInput
from .prediction_utils import prediction_label
from .utils import structural_similarity as default_structural_similarity

_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


@dataclass(frozen=True)
class _ContrastivityBatchContext:
    """Cached view of all usable labelled attribution vectors in a batch."""

    labeled_importance: List[tuple[Any, np.ndarray]]
    orig_to_filtered: Dict[int, int]
    unique_labels: List[Any]
    label_indices: Dict[Any, List[int]]


class ContrastivityEvaluator(MetricCapabilities):
    """
    Estimate target contrastivity by comparing attributions across classes.

    Adapted from the Random Logit metric by Sixt et al. (2020) / Quantus library.

    For each explanation we repeatedly sample a reference explanation predicted
    for a different class and compute SSIM similarity between the attribution
    vectors. Scores are inverted (1 - SSIM) so higher values indicate that
    explanations differ strongly across classes (i.e., high contrastivity).
    """

    per_instance = True
    requires_full_batch = True
    metric_names = ("contrastivity", "contrastivity_pairs")
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        pairs_per_instance: int = 3,
        normalise: bool = True,
        similarity_func: Optional[
            Union[str, Callable[[np.ndarray, np.ndarray], Optional[float]]]
        ] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        """
        Parameters
        ----------
        pairs_per_instance : int, optional
            How many off-class comparisons to sample per explanation.
        normalise : bool, optional
            Whether to L1-normalise attribution vectors before similarity.
        similarity_func : callable | str, optional
            Callable returning a similarity score given two vectors or a fully-qualified
            import path to such a function (default: SSIM util).
        random_state : int | None, optional
            Random seed for pairing instances across classes.
        """
        self.pairs_per_instance = max(1, int(pairs_per_instance))
        self.normalise = bool(normalise)
        self.similarity_func = self._resolve_similarity_func(similarity_func)
        self.random_state = random_state
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
        Compute the average contrastivity score over a batch of explanations.

        Parameters
        ----------
        model : Any
            Trained model associated with the explanations (unused placeholder).
        explanation_results : Dict[str, Any]
            Output dict from ``BaseExplainer.explain_dataset``.
        dataset : Any | None, optional
            Dataset reference (unused placeholder).
        explainer : Any | None, optional
            Explainer instance (unused placeholder).
        cache : Dict[str, Any] | None, optional
            Shared evaluator cache so per-instance execution can reuse the same
            filtered batch context across repeated calls.

        Returns
        -------
        Dict[str, float]
            Dictionary containing average contrastivity and number of evaluated pairs.
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
            Standardized evaluator payload containing batch explanations.

        Returns
        -------
        Dict[str, float]
            Averaged contrastivity and pair count (zeros if requirements unmet).
        """
        if metric_input.method not in self.supported_methods:
            return self._empty_result()

        explanations = metric_input.explanations
        if len(explanations) < 2:
            return self._empty_result()

        batch_context = self._prepare_batch_context(metric_input)
        if batch_context is None or len(batch_context.labeled_importance) < 2:
            return self._empty_result()

        if len(batch_context.unique_labels) < 2:
            self.logger.info("ContrastivityEvaluator skipped: only one label present.")
            return self._empty_result()

        if metric_input.explanation_idx is not None:
            filtered_idx = batch_context.orig_to_filtered.get(metric_input.explanation_idx)
            if filtered_idx is None:
                return self._empty_result()
            seed = (
                None
                if self.random_state is None
                else self.random_state + int(metric_input.explanation_idx)
            )
            rng = np.random.default_rng(seed)
            scores = self._contrastive_scores_for_index(filtered_idx, batch_context, rng)
            if not scores:
                return self._empty_result()
            return {
                "contrastivity": float(np.mean(scores)),
                "contrastivity_pairs": float(len(scores)),
            }

        rng = np.random.default_rng(self.random_state)
        contrastive_scores: List[float] = []

        for idx in range(len(batch_context.labeled_importance)):
            scores = self._contrastive_scores_for_index(idx, batch_context, rng)
            contrastive_scores.extend(scores)

        if not contrastive_scores:
            return self._empty_result()

        return {
            "contrastivity": float(np.mean(contrastive_scores)),
            "contrastivity_pairs": float(len(contrastive_scores)),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _contrastive_score(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
        """Return contrastivity score given two attribution vectors."""
        norm_a = self._normalise(vec_a)
        norm_b = self._normalise(vec_b)
        similarity = self.similarity_func(norm_a, norm_b)
        if similarity is None or np.isnan(similarity):
            return None
        score = 1.0 - float(similarity)
        return float(np.clip(score, 0.0, 1.0))

    def _normalise(self, vec: np.ndarray) -> np.ndarray:
        """Optional L1-normalisation for comparability across instances."""
        return prepare_attributions(
            vec,
            abs_attributions=False,
            normalise=self.normalise,
            normalise_mode="l1",
        )

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract attribution vector from explanation or metadata."""
        return extract_attribution_vector(
            explanation,
            logger=self.logger,
            log_prefix="ContrastivityEvaluator",
        )

    def _prediction_label(self, explanation: Dict[str, Any]) -> Any:
        """Derive the predicted label from explanation (uses prediction or proba)."""
        return prediction_label(explanation)

    def _empty_result(self) -> Dict[str, float]:
        return {"contrastivity": 0.0, "contrastivity_pairs": 0.0}

    def _prepare_batch_context(
        self,
        metric_input: MetricInput,
    ) -> Optional[_ContrastivityBatchContext]:
        """
        Collect valid (label, attribution) pairs once per batch so repeated
        per-instance calls can reuse the same filtered view of the explanations.
        """
        cache_bucket = metric_input.cache_bucket("contrastivity_context")
        cache_key = (
            id(metric_input.explanations),
            metric_input.method,
            self.normalise,
        )
        cached = cache_bucket.get(cache_key)
        if isinstance(cached, _ContrastivityBatchContext):
            return cached

        labeled_importance: List[tuple[Any, np.ndarray]] = []
        orig_to_filtered: Dict[int, int] = {}
        for orig_idx, explanation in metric_input.iter_explanations():
            importance = self._importance_vector(explanation)
            label = self._prediction_label(explanation)
            if importance is None or label is None:
                continue
            if importance.size == 0 or not np.all(np.isfinite(importance)):
                continue
            orig_to_filtered[orig_idx] = len(labeled_importance)
            labeled_importance.append((label, importance))

        if not labeled_importance:
            return None

        labels = [label for label, _ in labeled_importance]
        unique_labels = list({label for label in labels})
        label_indices: Dict[Any, List[int]] = {}
        for idx, label in enumerate(labels):
            label_indices.setdefault(label, []).append(idx)

        context = _ContrastivityBatchContext(
            labeled_importance=labeled_importance,
            orig_to_filtered=orig_to_filtered,
            unique_labels=unique_labels,
            label_indices=label_indices,
        )
        cache_bucket[cache_key] = context
        return context

    def _contrastive_scores_for_index(
        self,
        target_idx: int,
        batch_context: _ContrastivityBatchContext,
        rng: np.random.Generator,
    ) -> List[float]:
        """Sample contrastive scores anchored to a specific explanation index."""
        label, importance = batch_context.labeled_importance[target_idx]
        candidate_labels = [
            lbl
            for lbl in batch_context.unique_labels
            if lbl != label and batch_context.label_indices.get(lbl)
        ]
        if not candidate_labels:
            return []

        scores: List[float] = []
        for _ in range(self.pairs_per_instance):
            ref_label = rng.choice(candidate_labels)
            ref_idx = int(rng.choice(batch_context.label_indices[ref_label]))
            ref_importance = batch_context.labeled_importance[ref_idx][1]
            score = self._contrastive_score(importance, ref_importance)
            if score is not None:
                scores.append(score)
        return scores

    def _resolve_similarity_func(
        self,
        func: Optional[Union[str, Callable[[np.ndarray, np.ndarray], Optional[float]]]],
    ) -> Callable[[np.ndarray, np.ndarray], Optional[float]]:
        """Return a callable similarity function for attribution vectors."""
        if func is None:
            return default_structural_similarity
        if callable(func):
            return func
        if isinstance(func, str):
            module_path, _, attr = func.rpartition(".")
            if not module_path:
                raise ValueError(
                    f"similarity_func must be a callable or module path, got '{func}'."
                )
            module = import_module(module_path)
            attr_obj = getattr(module, attr)
            if not callable(attr_obj):
                raise TypeError(f"Resolved object '{func}' is not callable.")
            return attr_obj
        raise TypeError("similarity_func must be None, callable, or module path string.")
