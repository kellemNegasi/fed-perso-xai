"""Metric computation, predictions, and baseline comparison helpers."""

from .comparison import build_baseline_comparison, write_comparison_report
from .contracts import (
    ClientEvaluationReport,
    ExtensionEvaluationBundle,
    PredictiveEvaluationBundle,
    SplitEvaluationReport,
)
from .metrics import (
    aggregate_weighted_metrics,
    compute_classification_metrics,
    compute_pooled_classification_metrics,
    summarize_class_balance,
    summarize_probability_distribution,
    sweep_classification_thresholds,
)
from .predictions import (
    PredictionArtifact,
    build_prediction_artifact,
    load_prediction_artifact,
    save_prediction_artifact,
)

__all__ = [
    "ClientEvaluationReport",
    "ExtensionEvaluationBundle",
    "PredictionArtifact",
    "PredictiveEvaluationBundle",
    "SplitEvaluationReport",
    "aggregate_weighted_metrics",
    "build_baseline_comparison",
    "build_prediction_artifact",
    "compute_classification_metrics",
    "compute_pooled_classification_metrics",
    "load_prediction_artifact",
    "save_prediction_artifact",
    "summarize_class_balance",
    "summarize_probability_distribution",
    "sweep_classification_thresholds",
    "write_comparison_report",
]
