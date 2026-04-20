"""
Shared helper functions for evaluator modules (similarities, normalisation, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "coerce_metric_dict",
    "extract_metric_parameters",
    "safe_scalar",
    "structural_similarity",
    "value_at",
]


def structural_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
    """
    Lightweight SSIM variant for 1-D attribution vectors.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        Flattenable attribution vectors with identical shapes.

    Returns
    -------
    float | None
        Similarity score in [-1, 1]; None if the inputs are incompatible.
    """
    if vec_a.size == 0 or vec_b.size == 0 or vec_a.size != vec_b.size:
        return None

    a = vec_a.astype(float).ravel()
    b = vec_b.astype(float).ravel()
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None

    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    diff_a = a - mu_a
    diff_b = b - mu_b

    var_a = float(np.mean(diff_a * diff_a))
    var_b = float(np.mean(diff_b * diff_b))
    cov_ab = float(np.mean(diff_a * diff_b))

    data_range = max(np.max(a) - np.min(a), np.max(b) - np.min(b))
    if data_range < 1e-12:
        data_range = 1.0

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    denom_mean = mu_a**2 + mu_b**2 + c1
    denom_var = var_a + var_b + c2
    if denom_mean <= 0.0 or denom_var <= 0.0:
        if np.allclose(a, b):
            return 1.0
        return 0.0

    numerator = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    denominator = denom_mean * denom_var
    if denominator == 0.0:
        return 0.0

    ssim_value = numerator / denominator
    return float(np.clip(ssim_value, -1.0, 1.0))


def extract_metric_parameters(metric: Any) -> Dict[str, Any]:
    """Return the public JSON-able attributes of a metric instance."""
    params: Dict[str, Any] = {}
    attr_dict = getattr(metric, "__dict__", {})
    for key, value in attr_dict.items():
        if key.startswith("_"):
            continue
        if _is_jsonable(value):
            params[key] = value
    return params


def coerce_metric_dict(values: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Normalize a metric output mapping to floats, dropping invalid entries."""
    if not values:
        return {}
    coerced: Dict[str, float] = {}
    for key, value in values.items():
        if value is None:
            continue
        try:
            coerced[key] = float(value)
        except (TypeError, ValueError):
            continue
    return coerced


def value_at(sequence: Any, index: int) -> Any:
    """Safely fetch sequence[index], returning None for out-of-range access."""
    if sequence is None or index < 0:
        return None
    try:
        length = len(sequence)
    except TypeError:
        return None
    if index >= length:
        return None
    try:
        return sequence[index]
    except (IndexError, TypeError):
        return None


def safe_scalar(value: Any) -> Any:
    """Convert numpy scalar/array to a native scalar when possible."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_jsonable(value: Any) -> bool:
    """Internal helper to check whether a value can be serialized to JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_jsonable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) for k in value.keys()) and all(
            _is_jsonable(v) for v in value.values()
        )
    return False
