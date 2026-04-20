"""Metric-evaluation helpers shared across orchestration components."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_metric import MetricInput
from .utils import coerce_metric_dict

LOGGER = logging.getLogger(__name__)


DatasetMappingValue = Union[Tuple[int, Dict[str, Any]], List[Tuple[int, Dict[str, Any]]]]


@dataclass(frozen=True)
class MetricExecutionResult:
    """Structured output for one explainer-method metric run."""

    batch_metrics: Dict[str, float] = field(default_factory=dict)
    instance_metrics: Dict[int, Dict[int, Dict[str, float]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_metrics": self.batch_metrics,
            "instance_metrics": self.instance_metrics,
        }


def evaluate_metrics_for_method(
    *,
    metric_objs: Dict[str, Any],
    metric_caps: Dict[str, Dict[str, Any]],
    explainer: Any,
    expl_results: Dict[str, Any],
    dataset_mapping: Dict[int, DatasetMappingValue],
    model: Any,
    dataset: Any,
    method_label: str,
    log_progress: bool,
) -> MetricExecutionResult:
    """Execute metric evaluators for a single explainer method.

    Returns
    -------
    batch_metrics : Dict[str, float]
    instance_metrics : Dict[int, Dict[int, Dict[str, float]]]
        Mapping dataset_index -> explanation_index (local_idx) -> metric values.
    """
    batch_metrics: Dict[str, float] = {}
    instance_metrics: Dict[int, Dict[int, Dict[str, float]]] = {}
    if not metric_objs:
        return MetricExecutionResult(batch_metrics=batch_metrics, instance_metrics=instance_metrics)

    def _iter_entries(mapping_value: DatasetMappingValue):
        """Yield (local_idx, explanation) tuples regardless of single/list input."""
        if isinstance(mapping_value, list):
            for entry in mapping_value:
                yield entry
        else:
            yield mapping_value

    shared_cache: Dict[str, Any] = {}

    for metric_name, metric in metric_objs.items():
        caps = metric_caps[metric_name]
        if caps["per_instance"]:
            if log_progress:
                LOGGER.info(
                    "Running %s metric (per-instance) for %s", metric_name, method_label
                )
            for dataset_idx, mapping_value in dataset_mapping.items():
                for entry in _iter_entries(mapping_value):
                    try:
                        local_idx, _ = entry
                    except (TypeError, ValueError):
                        continue
                    payload = dict(expl_results)
                    payload["current_index"] = local_idx
                    out = evaluate_metric(
                        metric,
                        model=model,
                        explanation_results=payload,
                        dataset=dataset,
                        explainer=explainer,
                        cache=shared_cache,
                    )
                    values = coerce_metric_dict(out)
                    if not values:
                        continue
                    dataset_bucket = instance_metrics.setdefault(int(dataset_idx), {})
                    metrics_bucket = dataset_bucket.setdefault(int(local_idx), {})
                    metrics_bucket.update(values)
            continue

        if not caps["requires_full_batch"]:
            continue
        if log_progress:
            LOGGER.info("Running %s metric (batch) for %s", metric_name, method_label)
        out = evaluate_metric(
            metric,
            model=model,
            explanation_results=expl_results,
            dataset=dataset,
            explainer=explainer,
            cache=shared_cache,
        )
        batch_metrics.update(coerce_metric_dict(out))

    return MetricExecutionResult(batch_metrics=batch_metrics, instance_metrics=instance_metrics)


def evaluate_metric(
    metric: Any,
    *,
    model: Any,
    explanation_results: Dict[str, Any],
    dataset: Any | None = None,
    explainer: Any | None = None,
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate one metric using either the public ``evaluate`` API or ``_evaluate``.

    The concrete Perso-XAI metrics expose ``evaluate``. For the shared layer and
    tests we also accept a lightweight metric implementing ``_evaluate`` on top of
    ``MetricInput`` directly.
    """
    if hasattr(metric, "evaluate"):
        try:
            return metric.evaluate(
                model=model,
                explanation_results=explanation_results,
                dataset=dataset,
                explainer=explainer,
                cache=cache,
            )
        except TypeError:
            return metric.evaluate(
                model=model,
                explanation_results=explanation_results,
                dataset=dataset,
                explainer=explainer,
            )

    if hasattr(metric, "_evaluate"):
        metric_input = MetricInput.from_results(
            model=model,
            explanation_results=explanation_results,
            dataset=dataset,
            explainer=explainer,
            cache=cache,
        )
        return metric._evaluate(metric_input)

    raise TypeError("Metric object must expose evaluate(...) or _evaluate(metric_input).")
