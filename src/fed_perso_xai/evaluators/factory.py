from __future__ import annotations

import importlib
from typing import Any

from .registry import DEFAULT_METRIC_REGISTRY, MetricRegistry


def make_metric(name: str, *, registry: MetricRegistry | None = None) -> Any:
    """Instantiate one explanation-quality metric from the YAML-backed registry."""
    spec = (registry or DEFAULT_METRIC_REGISTRY).get(name)
    module_name = str(spec["module"])
    class_name = str(spec["class"])
    params = dict(spec.get("params", {}) or {})

    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except Exception as exc:
        raise NotImplementedError(
            f"Metric '{name}' is declared in configs/metrics.yml but its implementation "
            f"({module_name}.{class_name}) has not been ported into fed-perso-xai yet."
        ) from exc
    return cls(**params)


def metric_capabilities(metric: Any) -> dict[str, bool]:
    return {
        "per_instance": getattr(metric, "per_instance", True),
        "requires_full_batch": getattr(metric, "requires_full_batch", False),
    }
