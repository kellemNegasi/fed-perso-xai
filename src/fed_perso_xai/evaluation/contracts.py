"""Typed evaluation payloads shared by centralized and federated summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SplitEvaluationReport:
    """Machine-readable evaluation report for one split."""

    split_name: str
    provenance: dict[str, Any]
    class_balance: dict[str, Any]
    probability_summary: dict[str, float]
    metrics: dict[str, float]
    predictions_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "split_name": self.split_name,
            "provenance": self.provenance,
            "class_balance": self.class_balance,
            "probability_summary": self.probability_summary,
            "metrics": self.metrics,
        }
        if self.predictions_path is not None:
            payload["predictions_path"] = self.predictions_path
        return payload


@dataclass(frozen=True)
class ClientEvaluationReport:
    """Machine-readable evaluation report for one client."""

    client_id: int
    split_name: str
    num_examples: int
    class_balance: dict[str, Any]
    probability_summary: dict[str, float]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "split_name": self.split_name,
            "num_examples": self.num_examples,
            "class_balance": self.class_balance,
            "probability_summary": self.probability_summary,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class PredictiveEvaluationBundle:
    """Typed bundle for predictive split and per-client evaluation outputs."""

    splits: dict[str, SplitEvaluationReport]
    per_client: list[ClientEvaluationReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "splits": {
                section_name: report.to_dict()
                for section_name, report in self.splits.items()
            },
            "per_client": [report.to_dict() for report in self.per_client],
        }


@dataclass(frozen=True)
class ExtensionEvaluationBundle:
    """Reserved slots for later explanation and downstream stages."""

    explanation_metrics: dict[str, Any] = field(default_factory=dict)
    explanation_artifacts: dict[str, str] = field(default_factory=dict)
    downstream_metrics: dict[str, Any] = field(default_factory=dict)
    downstream_artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanations": {
                "metrics": self.explanation_metrics,
                "artifacts": self.explanation_artifacts,
            },
            "downstream": {
                "metrics": self.downstream_metrics,
                "artifacts": self.downstream_artifacts,
            },
        }
