"""Centralized-versus-federated comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_COMPARISON_ORDER = (
    "centralized_global_eval",
    "centralized_pooled_client_test",
    "federated_client_test_weighted",
    "federated_client_test_pooled",
)


def build_baseline_comparison(
    *,
    centralized_summary: dict[str, Any],
    federated_summary: dict[str, Any],
    centralized_manifest: dict[str, Any] | None = None,
    federated_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a predictive comparison report between centralized and federated runs."""

    centralized_eval = centralized_summary["evaluation"]
    federated_eval = federated_summary["evaluation"]
    split_sections = {
        "centralized_global_eval": centralized_eval["global_eval"],
        "centralized_pooled_client_test": centralized_eval.get("pooled_client_test"),
        "federated_client_test_weighted": federated_eval["client_test_weighted"],
        "federated_client_test_pooled": federated_eval["client_test_pooled"],
    }

    metric_names = _collect_metric_names(split_sections)
    metric_rows = []
    baseline_metrics = split_sections["centralized_global_eval"]["metrics"]
    for metric_name in metric_names:
        row = {"metric": metric_name}
        for section_name in DEFAULT_COMPARISON_ORDER:
            section = split_sections.get(section_name)
            row[section_name] = None if section is None else section["metrics"].get(metric_name)
        row["absolute_difference_weighted_vs_centralized"] = _safe_absolute_difference(
            row["federated_client_test_weighted"],
            row["centralized_global_eval"],
        )
        row["absolute_difference_pooled_vs_centralized"] = _safe_absolute_difference(
            row["federated_client_test_pooled"],
            row["centralized_global_eval"],
        )
        row["absolute_difference_centralized_pooled_vs_global_eval"] = _safe_absolute_difference(
            row["centralized_pooled_client_test"],
            row["centralized_global_eval"],
        )
        row["metric_availability_notes"] = _build_metric_availability_notes(
            metric_name=metric_name,
            row=row,
        )
        metric_rows.append(row)

    return {
        "comparison_version": "stage1_predictive_comparison_v2",
        "dataset_name": centralized_summary["dataset_name"],
        "centralized_run": centralized_summary["result_dir"],
        "federated_run": federated_summary["result_dir"],
        "source_references": {
            "centralized": {
                "run_directory": centralized_summary["result_dir"],
                "run_manifest_path": centralized_summary.get("run_manifest_path"),
                "run_manifest": centralized_manifest,
            },
            "federated": {
                "run_directory": federated_summary["result_dir"],
                "run_manifest_path": federated_summary.get("run_manifest_path"),
                "run_manifest": federated_manifest,
            },
        },
        "headline_metrics": {
            "centralized_global_holdout_metrics": split_sections["centralized_global_eval"]["metrics"],
            "federated_weighted_metrics": split_sections["federated_client_test_weighted"]["metrics"],
            "federated_pooled_metrics": split_sections["federated_client_test_pooled"]["metrics"],
        },
        "metric_schema": metric_names,
        "split_reports": split_sections,
        "predictive_metric_comparison": metric_rows,
        "federated_per_client_summary": _summarize_per_client_metrics(federated_eval["per_client"]),
        "federated_per_client_metrics": federated_eval["per_client"],
        "extension_points": {
            "future_explanation_metrics": "Add extra split reports or comparison tables keyed by the same run manifest and prediction artifacts.",
            "future_recommender_metrics": "Attach downstream comparison sections alongside predictive_metric_comparison without changing existing predictive artifacts.",
        },
        "provenance": {
            "centralized_run_id": centralized_summary.get("run_id"),
            "federated_run_id": federated_summary.get("run_id"),
            "centralized_metric_baseline": baseline_metrics,
            "federated_runtime": federated_summary.get("runtime"),
        },
    }


def write_comparison_report(path: Path, report: dict[str, Any]) -> Path:
    """Persist a comparison report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _collect_metric_names(split_sections: dict[str, dict[str, Any] | None]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for section_name in DEFAULT_COMPARISON_ORDER:
        section = split_sections.get(section_name)
        if section is None:
            continue
        for metric_name in section["metrics"]:
            if metric_name not in seen:
                seen.add(metric_name)
                ordered.append(metric_name)
    return ordered


def _summarize_per_client_metrics(per_client_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            metric_name
            for row in per_client_rows
            for metric_name in row.get("metrics", {})
        }
    )
    metric_summaries = {}
    for metric_name in metric_names:
        values = [row["metrics"][metric_name] for row in per_client_rows if metric_name in row["metrics"]]
        metric_summaries[metric_name] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
    client_sizes = [row["num_examples"] for row in per_client_rows]
    return {
        "num_clients": len(per_client_rows),
        "client_size_summary": {
            "min": min(client_sizes),
            "max": max(client_sizes),
            "mean": sum(client_sizes) / len(client_sizes),
        },
        "metric_summaries": metric_summaries,
    }


def _safe_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _safe_absolute_difference(left: Any, right: Any) -> float | None:
    delta = _safe_delta(left, right)
    if delta is None:
        return None
    return abs(delta)


def _build_metric_availability_notes(*, metric_name: str, row: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    label_map = {
        "centralized_global_eval": "centralized_global_holdout",
        "centralized_pooled_client_test": "centralized_pooled_client_test",
        "federated_client_test_weighted": "federated_client_test_weighted",
        "federated_client_test_pooled": "federated_client_test_pooled",
    }
    for section_name, label in label_map.items():
        if row.get(section_name) is None:
            notes.append(f"Metric '{metric_name}' is unavailable for {label}.")
    return notes
