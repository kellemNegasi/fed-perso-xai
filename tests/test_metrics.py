from __future__ import annotations

from fed_perso_xai.evaluation.metrics import aggregate_weighted_metrics


def test_weighted_metric_aggregation_smoke() -> None:
    aggregated = aggregate_weighted_metrics(
        [
            (10, {"loss": 0.4, "accuracy": 0.8, "precision": 0.75, "recall": 0.9, "f1": 0.82, "roc_auc": 0.88}),
            (30, {"loss": 0.2, "accuracy": 0.9, "precision": 0.91, "recall": 0.85, "f1": 0.88, "roc_auc": 0.93}),
        ]
    )
    assert round(aggregated["loss"], 4) == 0.25
    assert round(aggregated["accuracy"], 4) == 0.875
    assert aggregated["num_examples"] == 40.0
