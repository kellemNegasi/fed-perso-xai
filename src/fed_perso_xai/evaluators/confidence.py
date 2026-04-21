"""
Confidence metric via bootstrap stability of attribution vectors.

This evaluator resamples explanations by re-running the explainer with
different random seeds (for stochastic methods such as LIME/Kernel SHAP) or
with alternative baselines (for deterministic methods such as Integrated
Gradients). The resulting attribution samples are converted into feature-wise
confidence scores using percentile intervals, and then aggregated into a
single weighted score.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .attribution_utils import (
    FEATURE_METHOD_KEYS,
    extract_attribution_vector,
    extract_instance_vector,
)
from .base_metric import MetricCapabilities, MetricInput
from .perturbation import build_metric_rng


_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


class ConfidenceEvaluator(MetricCapabilities):
    """Bootstrap-style confidence estimator for local feature attributions."""

    per_instance = True
    requires_full_batch = False
    metric_names = ("confidence",)
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        n_resamples: int = 8,
        ci_percentile: float = 95.0,
        max_instances: int = 10,
        noise_scale: float = 0.01,
        random_baseline: bool = True,
        metric_key: str = "confidence",
        random_state: Optional[int] = 0,
    ) -> None:
        """
        Parameters
        ----------
        n_resamples : int, optional
            Total number of attribution samples used to estimate the confidence
            interval (includes the original explanation). Defaults to 8.
        ci_percentile : float, optional
            Central percentile mass for the interval (95 -> use 2.5/97.5 percent).
        max_instances : int, optional
            Maximum number of explanations to process when aggregating batches.
        noise_scale : float, optional
            Standard deviation for small Gaussian perturbations applied when a
            method lacks inherent randomness (used as a fallback).
        random_baseline : bool, optional
            For Integrated Gradients, draw random training baselines each run
            instead of the deterministic training mean.
        metric_key : str, optional
            Name of the aggregate score in the returned dictionary.
        random_state : int | None, optional
            Base RNG seed for reproducible bootstrap sampling.
        """
        self.n_resamples = max(2, int(n_resamples))
        self.ci_percentile = float(np.clip(ci_percentile, 50.0, 99.9))
        self.max_instances = max(1, int(max_instances))
        self.noise_scale = float(max(0.0, noise_scale))
        self.random_baseline = bool(random_baseline)
        self.metric_key = metric_key or "confidence"
        self.metric_names = (self.metric_key,)
        self.random_state = int(random_state) if random_state is not None else None
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
        Compute the confidence score for the provided explanations.

        Parameters
        ----------
        model : Any
            Unused placeholder for API parity (confidence only inspects explanations).
        explanation_results : Dict[str, Any]
            Output of ``BaseExplainer.explain_dataset`` containing the explanations.
        dataset : Any | None, optional
            Dataset reference (unused directly; explainer already contains it).
        explainer : Any | None, optional
            Explainer instance used to regenerate attribution samples.
        cache : Dict[str, Any] | None, optional
            Shared evaluator cache. Accepted for compatibility with the shared
            metric execution path.
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
    # Core computation
    # ------------------------------------------------------------------ #

    def _evaluate(self, metric_input: MetricInput) -> Dict[str, float]:
        if metric_input.method not in self.supported_methods:
            return self._empty_result()

        explanations = metric_input.explanations
        if not explanations:
            return self._empty_result()

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return self._empty_result()
            score = self._confidence_for_explanation(
                metric_input=metric_input,
                explanation_index=idx,
                explanation=explanations[idx],
            )
            return self._format_result(score)

        scores: List[float] = []
        limit = min(len(explanations), self.max_instances)
        for idx, explanation in metric_input.iter_explanations():
            if idx >= limit:
                break
            score = self._confidence_for_explanation(
                metric_input=metric_input,
                explanation_index=idx,
                explanation=explanation,
            )
            if score is not None:
                scores.append(score)

        aggregate = float(np.mean(scores)) if scores else 0.0
        return self._format_result(aggregate)

    def _empty_result(self) -> Dict[str, float]:
        return {self.metric_key: 0.0}

    def _confidence_for_explanation(
        self,
        metric_input: MetricInput,
        explanation_index: int,
        explanation: Dict[str, Any],
    ) -> Optional[float]:
        # Seed per explanation index so batch and per-instance evaluation stay aligned.
        rng = build_metric_rng(self.random_state, offset=explanation_index)
        samples = self._collect_attribution_samples(metric_input, explanation, rng)
        if samples is None:
            return None
        per_feature_conf, aggregate = self._confidence_from_samples(samples)

        # Store per-feature confidences alongside the aggregate for transparency.
        metadata = explanation.setdefault("metadata", {})
        if metadata.get("confidence_per_feature") is None:
            metadata["confidence_per_feature"] = per_feature_conf.tolist()
        return aggregate

    def _collect_attribution_samples(
        self,
        metric_input: MetricInput,
        explanation: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Optional[np.ndarray]:
        base_vec = self._feature_vector(explanation)
        if base_vec is None or base_vec.size == 0 or not np.all(np.isfinite(base_vec)):
            return None

        instance = self._instance_vector(explanation)
        if instance is None or not np.all(np.isfinite(instance)):
            return None

        samples: List[np.ndarray] = [base_vec]
        for _ in range(self.n_resamples - 1):
            rerun = self._rerun_explainer(metric_input, instance, rng)
            if rerun is None:
                continue
            if rerun.size != base_vec.size:
                self.logger.debug(
                    "ConfidenceEvaluator discarded rerun with mismatched attribution size: %s != %s",
                    rerun.size,
                    base_vec.size,
                )
                continue
            if not np.all(np.isfinite(rerun)):
                self.logger.debug(
                    "ConfidenceEvaluator discarded rerun with non-finite attributions."
                )
                continue
            samples.append(rerun)

        if len(samples) < 2:
            return None
        return np.vstack(samples)

    def _rerun_explainer(
        self,
        metric_input: MetricInput,
        instance: np.ndarray,
        rng: np.random.Generator,
    ) -> Optional[np.ndarray]:
        base_explainer = metric_input.explainer
        if base_explainer is None:
            return None

        clone = self._instantiate_clone(base_explainer)
        if clone is None:
            return None

        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        if hasattr(clone, "random_state"):
            setattr(clone, "random_state", seed)

        if not self._fit_clone(clone):
            return None

        # Integrated gradients: sample random baseline vectors if requested.
        method = metric_input.method
        if method in {"integrated_gradients", "integratedgradients"} and self.random_baseline:
            self._override_ig_baseline(clone, rng)

        rerun_instance = instance
        if method in {
            "shap",
            "causal_shap",
            "causalshap",
            "integrated_gradients",
            "integratedgradients",
        }:
            rerun_instance = self._maybe_noise_instance(instance, rng)

        try:
            result = clone.explain_instance(rerun_instance)
        except Exception as exc:
            self.logger.debug("ConfidenceEvaluator clone explain_instance failed: %s", exc)
            return None
        return self._feature_vector(result)

    # ------------------------------------------------------------------ #
    # Confidence aggregation helpers
    # ------------------------------------------------------------------ #

    def _confidence_from_samples(self, samples: np.ndarray) -> Tuple[np.ndarray, float]:
        lower_pct = (100.0 - self.ci_percentile) / 2.0
        upper_pct = 100.0 - lower_pct
        lower = np.percentile(samples, lower_pct, axis=0)
        upper = np.percentile(samples, upper_pct, axis=0)
        width = np.maximum(upper - lower, 0.0)

        mean_abs = np.mean(np.abs(samples), axis=0)
        denom = mean_abs + width + 1e-8
        per_feature_conf = 1.0 - (width / denom)
        per_feature_conf = np.clip(per_feature_conf, 0.0, 1.0)

        weights = mean_abs
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            aggregate = float(np.mean(per_feature_conf))
        else:
            aggregate = float(np.sum(per_feature_conf * (weights / weight_sum)))
        return per_feature_conf, aggregate

    def _format_result(self, score: Optional[float]) -> Dict[str, float]:
        return {self.metric_key: float(score) if score is not None else 0.0}

    # ------------------------------------------------------------------ #
    # Clone utilities
    # ------------------------------------------------------------------ #

    def _instantiate_clone(self, explainer: Any) -> Any | None:
        try:
            explainer_cls = explainer.__class__
            config = copy.deepcopy(getattr(explainer, "config", {}))
            model = getattr(explainer, "model", None)
            dataset = getattr(explainer, "dataset", None)
            if model is None or dataset is None:
                raise ValueError("explainer clone requires model and dataset.")
            return explainer_cls(config=config, model=model, dataset=dataset)
        except Exception as exc:
            self.logger.debug("ConfidenceEvaluator failed to clone explainer: %s", exc)

        # Fall back to deepcopy so simple stateless explainers can still participate.
        try:
            return copy.deepcopy(explainer)
        except Exception as exc:
            self.logger.debug("ConfidenceEvaluator deepcopy clone failed: %s", exc)
            return None

    def _fit_clone(self, clone: Any) -> bool:
        if not hasattr(clone, "fit"):
            return True

        dataset = getattr(clone, "dataset", None)
        if dataset is None:
            return True

        X_train = getattr(dataset, "X_train", None)
        y_train = getattr(dataset, "y_train", None)
        if X_train is None:
            return True

        try:
            clone.fit(X_train, y_train)
            return True
        except Exception as exc:
            self.logger.debug("ConfidenceEvaluator clone.fit failed: %s", exc)
            return False

    def _override_ig_baseline(self, clone: Any, rng: np.random.Generator) -> None:
        X_train = getattr(clone, "_X_train", None)
        if X_train is None:
            return
        X_arr = np.asarray(X_train)
        if X_arr.size == 0:
            return
        idx = int(rng.integers(0, len(X_arr)))
        sampled = X_arr[idx]
        if sampled.ndim > 1:
            sampled = sampled.reshape(-1)
        setattr(clone, "_train_mean", sampled)

    def _maybe_noise_instance(self, instance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.noise_scale <= 0.0:
            return instance
        noise = rng.normal(0.0, self.noise_scale, size=instance.shape)
        return instance + noise

    # ------------------------------------------------------------------ #
    # Vector extractors
    # ------------------------------------------------------------------ #

    def _feature_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        return extract_attribution_vector(
            explanation,
            logger=self.logger,
            log_prefix="ConfidenceEvaluator",
        )

    def _instance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        return extract_instance_vector(explanation)
