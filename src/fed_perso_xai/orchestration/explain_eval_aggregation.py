"""Aggregate shard-level explain/evaluate artifacts into analysis tables."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fed_perso_xai.orchestration.run_artifacts import resolve_federated_run_context
from fed_perso_xai.utils.config import ArtifactPaths
from fed_perso_xai.utils.provenance import current_utc_timestamp


@dataclass(frozen=True)
class ExplainEvalShardJob:
    """Resolved artifacts for one completed explain/evaluate shard job."""

    client_id: str
    shard_id: str
    explanations_path: Path
    metrics_path: Path
    done_marker_path: Path


def aggregate_explain_eval_results(
    *,
    run_id: str,
    selection_id: str,
    explainer_name: str,
    config_id: str,
    paths: ArtifactPaths | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Aggregate completed explain/evaluate shard outputs for one config."""

    _require_safe_segment(run_id, label="run_id")
    _require_safe_segment(selection_id, label="selection_id")
    _require_safe_segment(explainer_name, label="explainer_name")
    _require_safe_segment(config_id, label="config_id")

    artifact_paths = paths or ArtifactPaths()
    run_context = resolve_federated_run_context(paths=artifact_paths, run_id=run_id)
    shard_jobs, missing_jobs = _discover_shard_jobs(
        run_artifact_dir=run_context.run_artifact_dir,
        selection_id=selection_id,
        explainer_name=explainer_name,
        config_id=config_id,
    )
    if missing_jobs and not allow_partial:
        missing_preview = "; ".join(missing_jobs[:5])
        raise FileNotFoundError(
            "Cannot aggregate explain/evaluate results because some shard artifacts "
            f"are incomplete. Missing examples: {missing_preview}"
        )
    if not shard_jobs:
        raise FileNotFoundError(
            "No completed explain/evaluate shard jobs were found for "
            f"run_id={run_id!r}, selection={selection_id!r}, "
            f"explainer={explainer_name!r}, config={config_id!r}."
        )

    instance_metrics, job_metrics, metric_names = _collect_metric_rows(shard_jobs)
    explanation_rows = _collect_explanation_rows(shard_jobs)
    instance_metrics_frame = pd.DataFrame(instance_metrics)
    job_metrics_frame = pd.DataFrame(job_metrics)
    explanations_long_frame = pd.DataFrame(explanation_rows)
    feature_importance_frame = _build_feature_importance_by_client(explanations_long_frame)
    client_metric_summary_frame = _build_client_metric_summary(
        instance_metrics_frame,
        metric_names=metric_names,
    )
    divergence_summary_frame = _build_divergence_summary(feature_importance_frame)

    output_dir = (
        run_context.run_artifact_dir
        / "aggregations"
        / "explain_eval"
        / selection_id
        / explainer_name
        / config_id
    )
    artifacts = {
        "instance_metrics": output_dir / "instance_metrics.parquet",
        "job_metrics": output_dir / "job_metrics.parquet",
        "explanations_long": output_dir / "explanations_long.parquet",
        "feature_importance_by_client": output_dir / "feature_importance_by_client.parquet",
        "client_metric_summary": output_dir / "client_metric_summary.parquet",
        "divergence_summary": output_dir / "divergence_summary.parquet",
        "aggregation_metadata": output_dir / "aggregation_metadata.json",
    }
    _write_parquet_atomic(artifacts["instance_metrics"], instance_metrics_frame)
    _write_parquet_atomic(artifacts["job_metrics"], job_metrics_frame)
    _write_parquet_atomic(artifacts["explanations_long"], explanations_long_frame)
    _write_parquet_atomic(artifacts["feature_importance_by_client"], feature_importance_frame)
    _write_parquet_atomic(artifacts["client_metric_summary"], client_metric_summary_frame)
    _write_parquet_atomic(artifacts["divergence_summary"], divergence_summary_frame)

    client_count = int(instance_metrics_frame["client_id"].nunique()) if not instance_metrics_frame.empty else 0
    metadata = {
        "status": "aggregated",
        "run_id": run_id,
        "selection_id": selection_id,
        "explainer_name": explainer_name,
        "config_id": config_id,
        "generated_at": current_utc_timestamp(),
        "allow_partial": bool(allow_partial),
        "client_count": client_count,
        "shard_job_count": len(shard_jobs),
        "missing_job_count": len(missing_jobs),
        "instance_count": int(len(instance_metrics_frame)),
        "explanation_feature_row_count": int(len(explanations_long_frame)),
        "metric_names": sorted(metric_names),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "inputs": [
            {
                "client_id": job.client_id,
                "shard_id": job.shard_id,
                "explanations_path": str(job.explanations_path),
                "metrics_path": str(job.metrics_path),
                "done_marker_path": str(job.done_marker_path),
            }
            for job in shard_jobs
        ],
    }
    _write_json_atomic(artifacts["aggregation_metadata"], metadata)
    return metadata


def _discover_shard_jobs(
    *,
    run_artifact_dir: Path,
    selection_id: str,
    explainer_name: str,
    config_id: str,
) -> tuple[list[ExplainEvalShardJob], list[str]]:
    clients_root = run_artifact_dir / "clients"
    if not clients_root.exists():
        raise FileNotFoundError(f"Missing clients artifact directory: {clients_root}")

    shard_jobs: list[ExplainEvalShardJob] = []
    missing_jobs: list[str] = []
    for client_dir in sorted(path for path in clients_root.iterdir() if path.is_dir()):
        selection_dir = client_dir / "selections" / selection_id
        if not selection_dir.exists():
            continue
        shards_root = selection_dir / "shards"
        if not shards_root.exists():
            missing_jobs.append(f"{client_dir.name}: missing shards directory {shards_root}")
            continue
        for shard_dir in sorted(path for path in shards_root.iterdir() if path.is_dir()):
            metrics_path = shard_dir / "metrics_results" / explainer_name / f"{config_id}.json"
            explanations_path = (
                shard_dir / "detailed_explanations" / explainer_name / f"{config_id}.parquet"
            )
            done_marker_path = shard_dir / "_status" / f"{config_id}.done"
            missing = [
                str(path)
                for path in (metrics_path, explanations_path, done_marker_path)
                if not path.exists()
            ]
            if missing:
                missing_jobs.append(f"{client_dir.name}/{shard_dir.name}: missing {', '.join(missing)}")
                continue
            shard_jobs.append(
                ExplainEvalShardJob(
                    client_id=client_dir.name,
                    shard_id=shard_dir.name,
                    explanations_path=explanations_path,
                    metrics_path=metrics_path,
                    done_marker_path=done_marker_path,
                )
            )
    return shard_jobs, missing_jobs


def _collect_metric_rows(
    shard_jobs: list[ExplainEvalShardJob],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    instance_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    metric_names: set[str] = set()
    for job in shard_jobs:
        payload = json.loads(job.metrics_path.read_text(encoding="utf-8"))
        job_row = {
            "run_id": str(payload.get("run_id", "")),
            "client_id": str(payload.get("client_id", job.client_id)),
            "split": str(payload.get("split", "")),
            "selection_id": str(payload.get("selection_id", "")),
            "shard_id": str(payload.get("shard_id", job.shard_id)),
            "explainer_name": str(payload.get("explainer_name", "")),
            "config_id": str(payload.get("config_id", "")),
            "selected_instance_count": int(
                (payload.get("shard_metadata") or {}).get(
                    "selected_instance_count",
                    len(payload.get("per_instance_results") or []),
                )
            ),
            "metrics_path": str(job.metrics_path),
            "explanations_path": str(job.explanations_path),
        }
        for key, value in (payload.get("batch_metrics") or {}).items():
            job_row[f"batch__{key}"] = _coerce_optional_float(value)
        job_rows.append(job_row)

        for item in payload.get("per_instance_results") or []:
            row = {
                "run_id": str(payload.get("run_id", "")),
                "client_id": str(payload.get("client_id", job.client_id)),
                "split": str(payload.get("split", "")),
                "selection_id": str(payload.get("selection_id", "")),
                "shard_id": str(payload.get("shard_id", job.shard_id)),
                "explainer_name": str(payload.get("explainer_name", "")),
                "config_id": str(payload.get("config_id", "")),
                "instance_id": str(item.get("instance_id", "")),
                "dataset_index": _coerce_optional_int(item.get("dataset_index")),
                "instance_index_within_job": _coerce_optional_int(
                    item.get("instance_index_within_job")
                ),
                "true_label": item.get("true_label"),
                "prediction": item.get("prediction"),
                "explained_class": item.get("explained_class"),
                "method_variant": item.get("method_variant"),
            }
            for key, value in (item.get("metrics") or {}).items():
                metric_names.add(str(key))
                row[str(key)] = _coerce_optional_float(value)
            instance_rows.append(row)
    return instance_rows, job_rows, metric_names


def _collect_explanation_rows(shard_jobs: list[ExplainEvalShardJob]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in shard_jobs:
        frame = pd.read_parquet(job.explanations_path)
        for _, item in frame.iterrows():
            attributions = _load_json_list(
                item.get("attributions_json"),
                path=job.explanations_path,
                column="attributions_json",
            )
            feature_names = _load_json_list(
                item.get("feature_names_json"),
                path=job.explanations_path,
                column="feature_names_json",
            )
            instance_values = _load_json_list(
                item.get("instance_json"),
                path=job.explanations_path,
                column="instance_json",
            )
            if len(attributions) != len(feature_names):
                raise ValueError(
                    f"Attribution count and feature-name count differ in {job.explanations_path} "
                    f"for instance_id={item.get('instance_id')!r}."
                )
            if len(instance_values) != len(feature_names):
                instance_values = [None] * len(feature_names)
            base_row = {
                "run_id": item.get("run_id"),
                "client_id": item.get("client_id"),
                "split": item.get("split"),
                "selection_id": item.get("selection_id"),
                "shard_id": item.get("shard_id"),
                "explainer_name": item.get("explainer_name"),
                "config_id": item.get("config_id"),
                "method_variant": item.get("method_variant"),
                "instance_id": item.get("instance_id"),
                "dataset_index": _coerce_optional_int(item.get("dataset_index")),
                "instance_index_within_job": _coerce_optional_int(
                    item.get("instance_index_within_job")
                ),
                "true_label": item.get("true_label"),
                "prediction": item.get("prediction"),
                "explained_class": item.get("explained_class"),
                "prediction_proba_json": item.get("prediction_proba_json"),
                "generation_time_seconds": _coerce_optional_float(
                    item.get("generation_time_seconds")
                ),
                "generated_at": item.get("generated_at"),
            }
            for feature_index, (feature_name, attribution, instance_value) in enumerate(
                zip(feature_names, attributions, instance_values, strict=True)
            ):
                attribution_value = _coerce_optional_float(attribution)
                rows.append(
                    {
                        **base_row,
                        "feature_index": int(feature_index),
                        "feature_name": str(feature_name),
                        "attribution": attribution_value,
                        "abs_attribution": (
                            abs(attribution_value) if attribution_value is not None else None
                        ),
                        "instance_value": _coerce_optional_float(instance_value),
                    }
                )
    return rows


def _build_feature_importance_by_client(explanations_long: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_id",
        "selection_id",
        "explainer_name",
        "config_id",
        "client_id",
        "feature_index",
        "feature_name",
        "instance_count",
        "mean_attribution",
        "std_attribution",
        "mean_abs_attribution",
        "max_abs_attribution",
        "positive_fraction",
        "negative_fraction",
    ]
    if explanations_long.empty:
        return pd.DataFrame(columns=columns)

    group_cols = [
        "run_id",
        "selection_id",
        "explainer_name",
        "config_id",
        "client_id",
        "feature_index",
        "feature_name",
    ]
    grouped = explanations_long.groupby(group_cols, dropna=False)
    frame = grouped.agg(
        instance_count=("attribution", "size"),
        mean_attribution=("attribution", "mean"),
        std_attribution=("attribution", "std"),
        mean_abs_attribution=("abs_attribution", "mean"),
        max_abs_attribution=("abs_attribution", "max"),
        positive_fraction=("attribution", lambda values: float((values > 0).mean())),
        negative_fraction=("attribution", lambda values: float((values < 0).mean())),
    ).reset_index()
    return frame[columns]


def _build_client_metric_summary(
    instance_metrics: pd.DataFrame,
    *,
    metric_names: set[str],
) -> pd.DataFrame:
    id_cols = ["run_id", "selection_id", "explainer_name", "config_id", "client_id"]
    if instance_metrics.empty:
        return pd.DataFrame(columns=[*id_cols, "instance_count"])

    metric_cols = [name for name in sorted(metric_names) if name in instance_metrics.columns]
    count_frame = instance_metrics.groupby(id_cols, dropna=False).size().reset_index(name="instance_count")
    if not metric_cols:
        return count_frame

    summary = instance_metrics.groupby(id_cols, dropna=False)[metric_cols].agg(
        ["mean", "std", "median", "min", "max"]
    )
    summary.columns = [f"{metric}__{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    return count_frame.merge(summary, on=id_cols, how="left")


def _build_divergence_summary(feature_importance: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_id",
        "selection_id",
        "explainer_name",
        "config_id",
        "client_a",
        "client_b",
        "feature_count",
        "mean_abs_cosine_distance",
        "mean_abs_jensen_shannon",
        "signed_cosine_distance",
    ]
    if feature_importance.empty:
        return pd.DataFrame(columns=columns)

    key_cols = ["run_id", "selection_id", "explainer_name", "config_id"]
    rows: list[dict[str, Any]] = []
    for key_values, group in feature_importance.groupby(key_cols, dropna=False):
        abs_pivot = group.pivot_table(
            index="client_id",
            columns="feature_name",
            values="mean_abs_attribution",
            fill_value=0.0,
            aggfunc="mean",
        )
        signed_pivot = group.pivot_table(
            index="client_id",
            columns="feature_name",
            values="mean_attribution",
            fill_value=0.0,
            aggfunc="mean",
        )
        clients = list(abs_pivot.index)
        for client_a, client_b in combinations(clients, 2):
            abs_a = abs_pivot.loc[client_a].to_numpy(dtype=float)
            abs_b = abs_pivot.loc[client_b].to_numpy(dtype=float)
            signed_a = signed_pivot.loc[client_a].to_numpy(dtype=float)
            signed_b = signed_pivot.loc[client_b].to_numpy(dtype=float)
            rows.append(
                {
                    "run_id": key_values[0],
                    "selection_id": key_values[1],
                    "explainer_name": key_values[2],
                    "config_id": key_values[3],
                    "client_a": client_a,
                    "client_b": client_b,
                    "feature_count": int(abs_pivot.shape[1]),
                    "mean_abs_cosine_distance": _cosine_distance(abs_a, abs_b),
                    "mean_abs_jensen_shannon": _jensen_shannon_divergence(abs_a, abs_b),
                    "signed_cosine_distance": _cosine_distance(signed_a, signed_b),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _require_safe_segment(value: str, *, label: str) -> None:
    segment = str(value)
    if not segment or segment in {".", ".."} or "/" in segment or "\\" in segment:
        raise ValueError(f"{label} must be a single non-empty path segment.")


def _load_json_list(value: Any, *, path: Path, column: str) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON list in {path} column {column}.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list in {path} column {column}.")
    return parsed


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def _jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left_prob = _normalise_nonnegative(left)
    right_prob = _normalise_nonnegative(right)
    midpoint = 0.5 * (left_prob + right_prob)
    return float(
        0.5 * _kl_divergence(left_prob, midpoint)
        + 0.5 * _kl_divergence(right_prob, midpoint)
    )


def _normalise_nonnegative(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
    if clipped.size == 0:
        return clipped
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(clipped.shape, 1.0 / clipped.size)
    return clipped / total


def _kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
    epsilon = 1e-12
    safe_left = np.clip(left, epsilon, None)
    safe_right = np.clip(right, epsilon, None)
    return float(np.sum(safe_left * np.log(safe_left / safe_right)))


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Aggregating explain/evaluate artifacts requires 'pyarrow'. Install the project "
            "dependencies with Parquet support before running aggregate-explain-eval."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)
