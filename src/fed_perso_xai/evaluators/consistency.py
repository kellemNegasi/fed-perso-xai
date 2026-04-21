"""
Consistency evaluator adapted from the Quantus metric
(https://github.com/understandable-machine-intelligence-lab/Quantus).

Discretises attribution vectors and measures how often instances that receive
the same discretised explanation share the same predicted class, mirroring
Dasgupta et al. (ICML 2022).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np

from .attribution_utils import FEATURE_METHOD_KEYS, extract_attribution_vector
from .base_metric import MetricCapabilities, MetricInput
from .prediction_utils import prediction_label

_FEATURE_METHOD_KEYS = FEATURE_METHOD_KEYS


def _top_n_sign(values: np.ndarray, *, n: int = 5) -> tuple[tuple[int, int], ...]:
    """
    Default discretiser: keep the sign of the ``n`` largest-magnitude attribution
    components and convert them into a comparable token. This prefers comparing
    the most influential features first rather than the first ``n`` indices.

    Ties are broken by feature index so the token remains deterministic.
    """
    if values.size == 0:
        return ()
    n = max(1, min(n, values.size))
    order = np.lexsort((np.arange(values.size), -np.abs(values)))
    top_indices = order[:n]
    signs = np.sign(values[top_indices]).astype(int)
    return tuple(
        (int(feature_idx), int(sign))
        for feature_idx, sign in zip(top_indices.tolist(), signs.tolist())
    )


@dataclass(frozen=True)
class _ConsistencyBatchContext:
    """Cached per-instance consistency scores for one explanation batch."""

    scores: np.ndarray


DiscretiseFunc = Union[str, Callable[[np.ndarray], Any]]


class ConsistencyEvaluator(MetricCapabilities):
    """
    Probability that two instances sharing the same discretised explanation also
    share the same predicted class (higher is better).

    Parameters
    ----------
    discretise_func : callable | str, optional
        Function that maps an attribution vector to a comparable label. Defaults
        to ``top_n_sign`` from Quantus. A dotted module path is also accepted so
        YAML config entries can swap in custom discretisers.
    discretise_kwargs : dict, optional
        Extra keyword arguments forwarded to ``discretise_func``.
    """

    per_instance = True
    requires_full_batch = True
    metric_names = ("consistency",)
    supported_methods = tuple(_FEATURE_METHOD_KEYS)

    def __init__(
        self,
        *,
        discretise_func: Optional[DiscretiseFunc] = None,
        discretise_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.discretise_func = discretise_func
        self.discretise_kwargs = discretise_kwargs or {}
        self._resolved_discretise_func = self._resolve_discretise_func(discretise_func)
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

        batch_context = self._prepare_batch_context(metric_input)
        if batch_context is None:
            return self._empty_result()

        scores = batch_context.scores
        if metric_input.explanation_idx is not None:
            idx = metric_input.explanation_idx
            if 0 <= idx < len(scores):
                return {"consistency": float(scores[idx])}
            return self._empty_result()

        valid = scores[np.isfinite(scores)]
        if valid.size == 0:
            return self._empty_result()

        return {"consistency": float(np.mean(valid))}

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _empty_result(self) -> Dict[str, float]:
        return {"consistency": 0.0}

    def _prepare_batch_context(
        self,
        metric_input: MetricInput,
    ) -> Optional[_ConsistencyBatchContext]:
        """
        Cache per-instance consistency scores so the execution layer can reuse
        the same cross-instance grouping across repeated per-instance calls.
        """
        cache_bucket = metric_input.cache_bucket("consistency_context")
        cache_key = (
            id(metric_input.explanations),
            metric_input.method,
            repr(self.discretise_func),
            repr(sorted(self.discretise_kwargs.items())),
        )
        cached = cache_bucket.get(cache_key)
        if isinstance(cached, _ConsistencyBatchContext):
            return cached

        scores = self._consistency_scores(metric_input.explanations)
        if scores is None:
            return None

        context = _ConsistencyBatchContext(scores=scores)
        cache_bucket[cache_key] = context
        return context

    def _consistency_scores(self, explanations: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
        tokens: list[Any] = []
        pred_classes: list[Any] = []
        for explanation in explanations:
            importance = self._importance_vector(explanation)
            label = self._prediction_label(explanation)
            if importance is None or label is None:
                tokens.append(None)
                pred_classes.append(None)
                continue
            if importance.size == 0 or not np.all(np.isfinite(importance)):
                tokens.append(None)
                pred_classes.append(None)
                continue
            discretised = self._discretise(importance)
            tokens.append(self._canonicalise_token(discretised))
            pred_classes.append(label)

        if not any(token is not None for token in tokens):
            return None

        scores = np.zeros(len(explanations), dtype=float)
        groups: dict[Any, list[int]] = {}
        for idx, token in enumerate(tokens):
            if token is None or pred_classes[idx] is None:
                continue
            groups.setdefault(token, []).append(idx)

        # Equivalent discretised explanations define the comparison sets.
        for indices in groups.values():
            if len(indices) <= 1:
                continue
            class_counts: dict[Any, int] = {}
            for idx in indices:
                label = pred_classes[idx]
                class_counts[label] = int(class_counts.get(label, 0)) + 1
            denom = len(indices) - 1
            for idx in indices:
                label = pred_classes[idx]
                scores[idx] = float((class_counts[label] - 1) / denom) if denom > 0 else 0.0
        return scores

    def _discretise(self, importance: np.ndarray) -> Any:
        func = self._resolved_discretise_func or _top_n_sign
        kwargs = dict(self.discretise_kwargs)
        if "n" in kwargs:
            original = kwargs["n"]
            kwargs["n"] = max(1, min(int(kwargs["n"]), importance.size))
            if kwargs["n"] != original:
                self.logger.debug(
                    "ConsistencyEvaluator clamped discretiser 'n' from %s to %s to match feature count.",
                    original,
                    kwargs["n"],
                )
        try:
            return func(importance, **kwargs)
        except TypeError:
            self.logger.debug(
                "ConsistencyEvaluator discretise_func did not accept kwargs %s; retrying without.",
                kwargs,
            )
            return func(importance)
        except Exception as exc:
            self.logger.debug("ConsistencyEvaluator discretise_func failed: %s", exc)
            return 0

    def _importance_vector(self, explanation: Dict[str, Any]) -> Optional[np.ndarray]:
        arr = extract_attribution_vector(
            explanation,
            logger=self.logger,
            log_prefix="ConsistencyEvaluator",
        )
        if arr is not None:
            return arr
        self.logger.debug("ConsistencyEvaluator missing importance vector in explanation")
        return None

    def _prediction_label(self, explanation: Dict[str, Any]) -> Any:
        label = prediction_label(explanation)
        if label is not None:
            return label

        prediction = explanation.get("prediction")
        if prediction is not None:
            self.logger.debug(
                "ConsistencyEvaluator could not coerce prediction to label: %s",
                prediction,
            )
            return None

        proba = explanation.get("prediction_proba")
        if proba is None:
            self.logger.debug("ConsistencyEvaluator missing prediction/prediction_proba in explanation")
            return None

        arr = np.asarray(proba).ravel()
        if arr.size == 0:
            self.logger.debug("ConsistencyEvaluator received empty prediction_proba array")
            return None
        return None

    def _resolve_discretise_func(
        self,
        func: Optional[DiscretiseFunc],
    ) -> Optional[Callable[[np.ndarray], Any]]:
        """Return a callable discretiser for attribution vectors."""
        if func is None:
            return None
        if callable(func):
            return func
        if isinstance(func, str):
            module_path, _, attr = func.rpartition(".")
            if not module_path:
                raise ValueError(
                    f"discretise_func must be a callable or module path, got '{func}'."
                )
            module = import_module(module_path)
            attr_obj = getattr(module, attr)
            if not callable(attr_obj):
                raise TypeError(f"Resolved object '{func}' is not callable.")
            return attr_obj
        raise TypeError("discretise_func must be None, callable, or module path string.")

    def _canonicalise_token(self, token: Any) -> Any:
        """
        Convert custom discretiser outputs into deterministic, hashable tokens.

        The original Perso-XAI port used Python's ``hash(...)`` over raw bytes,
        which depends on process-level hash randomisation. We keep the same
        equivalence semantics but preserve the token structure itself so tests and
        cross-run comparisons stay deterministic.
        """
        if isinstance(token, np.ndarray):
            return tuple(self._canonicalise_token(value) for value in token.tolist())
        if isinstance(token, (list, tuple)):
            return tuple(self._canonicalise_token(value) for value in token)
        if isinstance(token, dict):
            return tuple(
                sorted(
                    (self._canonicalise_token(key), self._canonicalise_token(value))
                    for key, value in token.items()
                )
            )
        if isinstance(token, np.generic):
            return token.item()
        return token
