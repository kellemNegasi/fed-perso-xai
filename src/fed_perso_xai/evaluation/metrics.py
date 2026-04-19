"""Classification metrics for local and aggregated federated evaluation."""

from __future__ import annotations

from math import isnan
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_NAMES = ("loss", "accuracy", "precision", "recall", "f1", "roc_auc")
DEFAULT_CLASSIFICATION_THRESHOLD = 0.35


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loss: float,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> dict[str, float]:
    """Compute per-client binary classification metrics."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float("nan"),
        "threshold": float(threshold),
    }
    if np.unique(y_true).shape[0] > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def compute_pooled_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    loss: float,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> dict[str, float]:
    """Compute one global set of metrics from pooled predictions."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        loss=loss,
        threshold=threshold,
    )
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "num_examples": float(y_true.shape[0]),
            "positive_class_rate": float(np.mean(y_true)),
            "predicted_positive_rate": float(np.mean(y_pred)),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }
    )
    return metrics


def summarize_class_balance(y_true: np.ndarray) -> dict[str, Any]:
    """Return a JSON-friendly class-balance summary."""

    y_true = np.asarray(y_true, dtype=np.int64)
    unique, counts = np.unique(y_true, return_counts=True)
    count_map = {str(int(label)): int(count) for label, count in zip(unique, counts, strict=False)}
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    total = int(y_true.shape[0])
    return {
        "num_examples": total,
        "counts": count_map,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": float(positive_count / total) if total else float("nan"),
    }


def summarize_probability_distribution(y_prob: np.ndarray) -> dict[str, float]:
    """Return summary statistics for predicted probabilities."""

    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(np.mean(y_prob)),
        "std": float(np.std(y_prob)),
        "min": float(np.min(y_prob)),
        "max": float(np.max(y_prob)),
        "p05": float(np.quantile(y_prob, 0.05)),
        "p25": float(np.quantile(y_prob, 0.25)),
        "p50": float(np.quantile(y_prob, 0.50)),
        "p75": float(np.quantile(y_prob, 0.75)),
        "p95": float(np.quantile(y_prob, 0.95)),
    }


def sweep_classification_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate pooled metrics across a range of probability thresholds."""

    if thresholds is None:
        thresholds = [round(value, 2) for value in np.linspace(0.05, 0.95, 19)]
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        rows.append(
            compute_pooled_classification_metrics(
                y_true=y_true,
                y_prob=y_prob,
                loss=float("nan"),
                threshold=threshold,
            )
        )
    best_by_f1 = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    return {
        "rows": rows,
        "best_by_f1": best_by_f1,
    }


def aggregate_weighted_metrics(
    metric_rows: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    """Aggregate metrics across clients using sample-count weighting."""

    if not metric_rows:
        raise ValueError("metric_rows must not be empty.")

    aggregated: dict[str, float] = {}
    total_examples = float(sum(num_examples for num_examples, _ in metric_rows))
    metric_names = sorted({key for _, metrics in metric_rows for key in metrics})
    for metric_name in metric_names:
        numerator = 0.0
        denominator = 0.0
        for num_examples, metrics in metric_rows:
            if metric_name not in metrics:
                continue
            value = float(metrics[metric_name])
            if isnan(value):
                continue
            numerator += num_examples * value
            denominator += num_examples
        aggregated[metric_name] = float(numerator / denominator) if denominator else float("nan")
    aggregated["num_examples"] = total_examples
    return aggregated


def build_metrics_summary(
    *,
    per_client: list[dict[str, Any]],
    aggregated: dict[str, float],
) -> dict[str, Any]:
    """Build a machine-readable evaluation summary payload."""

    return {
        "per_client": per_client,
        "aggregated_weighted": aggregated,
    }
