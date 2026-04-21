"""
Relative stability metrics adapted from Quantus
(https://github.com/understandable-machine-intelligence-lab/Quantus).

This module currently implements Relative Input Stability (RIS) from Agarwal
et al. (2022): perturb the input slightly, recompute attributions, and report
the maximum ratio between relative attribution change and relative input change.
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
from .perturbation import add_scaled_gaussian_noise, build_metric_rng

_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


class RelativeInputStabilityEvaluator(MetricCapabilities):
    """
    Estimate how sensitive an explanation is relative to input perturbations.

    For each instance, RIS samples small input perturbations, reruns the
    explainer, and computes:

        || (e(x) - e(x')) / e(x) || / max(|| (x - x') / x ||, eps_min)

    Higher scores imply the attribution changes faster than the input itself.
    """

    supported_methods = tuple(_FEATURE_METHOD_KEYS)
    metric_names = ("relative_input_stability",)
    per_instance = True
    requires_full_batch = False

    def __init__(
        self,
        *,
        metric_key: str = "relative_input_stability",
        max_instances: int = 5,
        num_samples: int = 10,
        noise_scale: float = 0.01,
        eps_min: float = 1e-6,
        bounded: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        self.metric_key = metric_key or "relative_input_stability"
        self.max_instances = max(1, int(max_instances))
        self.num_samples = max(1, int(num_samples))
        self.noise_scale = float(max(0.0, noise_scale))
        self.eps_min = float(max(1e-12, eps_min))
        self.bounded = bool(bounded)
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

        if metric_input.explainer is None or not hasattr(metric_input.explainer, "explain_instance"):
            self.logger.debug(
                "RelativeInputStabilityEvaluator requires an explainer instance to rerun perturbations."
            )
            return self._empty_result()

        std_vec = dataset_feature_std(metric_input.dataset, explanations[0])

        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if not (0 <= idx < len(explanations)):
                return self._empty_result()
            rng = build_metric_rng(self.random_state, offset=idx)
            score = self._ris_score(
                explanation=explanations[idx],
                explainer=metric_input.explainer,
                feature_std=std_vec,
                rng=rng,
            )
            return self._format_result(score)

        rng = build_metric_rng(self.random_state)
        scores: List[float] = []
        n_eval = min(self.max_instances, len(explanations))

        for i in range(n_eval):
            score = self._ris_score(
                explanation=explanations[i],
                explainer=metric_input.explainer,
                feature_std=std_vec,
                rng=rng,
            )
            if score is not None and np.isfinite(score):
                scores.append(float(score))

        return self._format_result(float(np.mean(scores)) if scores else None)

    # ------------------------------------------------------------------ #
    # RIS helpers                                                        #
    # ------------------------------------------------------------------ #

    def _empty_result(self) -> Dict[str, float]:
        return {self.metric_key: 0.0}

    def _format_result(self, score: Optional[float]) -> Dict[str, float]:
        return {self.metric_key: float(score) if score is not None else 0.0}

    def _ris_score(
        self,
        explanation: Dict[str, Any],
        explainer: Any,
        feature_std: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> Optional[float]:
        instance = self._instance_vector(explanation)
        importance = self._importance_vector(explanation)
        if instance is None or importance is None or importance.size == 0:
            return None
        if not np.all(np.isfinite(instance)) or not np.all(np.isfinite(importance)):
            self.logger.debug("RelativeInputStabilityEvaluator received non-finite inputs.")
            return None

        baseline = np.abs(instance) + self.eps_min
        attr_base = np.abs(importance) + self.eps_min
        ratios: List[float] = []

        for _ in range(self.num_samples):
            perturbed_instance = add_scaled_gaussian_noise(
                instance,
                feature_std=feature_std,
                noise_scale=self.noise_scale,
                rng=rng,
            )
            input_delta = instance - perturbed_instance

            perturbed_importance = self._rerun_explainer(
                explainer=explainer,
                perturbed_instance=perturbed_instance,
            )
            if perturbed_importance is None or perturbed_importance.size != importance.size:
                if perturbed_importance is not None:
                    self.logger.debug(
                        "RelativeInputStabilityEvaluator discarded rerun with mismatched attribution size: %s != %s",
                        perturbed_importance.size,
                        importance.size,
                    )
                continue
            if not np.all(np.isfinite(perturbed_importance)):
                self.logger.debug(
                    "RelativeInputStabilityEvaluator discarded rerun with non-finite attributions."
                )
                continue

            rel_attr = np.linalg.norm((importance - perturbed_importance) / attr_base, ord=2)
            denom = np.linalg.norm(input_delta / baseline, ord=2)
            denom = max(float(denom), self.eps_min)
            ratio = float(rel_attr / denom)
            if np.isfinite(ratio):
                ratios.append(ratio)

        if not ratios:
            return None

        score = max(ratios)
        if not self.bounded:
            return score

        score = max(0.0, float(score))
        return score / (1.0 + score)  # to prevent unbounded growth, map to [0, 1).

    def _rerun_explainer(
        self,
        explainer: Any,
        perturbed_instance: np.ndarray,
    ) -> Optional[np.ndarray]:
        try:
            result = explainer.explain_instance(perturbed_instance)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.debug(
                "RelativeInputStabilityEvaluator failed to re-explain perturbed sample: %s",
                exc,
            )
            return None
        importance = self._importance_vector(result)
        if importance is None:
            self.logger.debug("RelativeInputStabilityEvaluator explanation missing attributions after rerun.")
        return importance

    def _instance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        candidate = extract_instance_vector(explanation)
        if candidate is None:
            self.logger.debug("RelativeInputStabilityEvaluator missing instance in explanation.")
        return candidate

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        importance = extract_attribution_vector(
            explanation,
            logger=self.logger,
            log_prefix="RelativeInputStabilityEvaluator",
        )
        if importance is None:
            self.logger.debug("RelativeInputStabilityEvaluator missing attribution vector in explanation.")
        return importance
