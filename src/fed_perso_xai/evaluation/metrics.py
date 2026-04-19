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
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5


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
            "true_positive_rate": float(np.mean(y_true)),
            "predicted_positive_rate": float(np.mean(y_pred)),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }
    )
    return metrics


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
