"""Prepare client-local recommender context features from explain/eval artifacts."""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from fed_perso_xai.data.serialization import load_client_datasets
from fed_perso_xai.explainers import (
    DEFAULT_EXPLAINER_REGISTRY,
    build_explainer_config_registry,
    resolve_explainer_config,
)
from fed_perso_xai.orchestration.run_artifacts import resolve_federated_run_context
from fed_perso_xai.utils.config import ArtifactPaths
from fed_perso_xai.utils.provenance import current_utc_timestamp


IGNORED_PARETO_METRICS = {
    "completeness_drop",
    "completeness_random_drop",
    "contrastivity_pairs",
}

LOWER_IS_BETTER_METRICS = {
    "infidelity",
    "non_sensitivity_violation_fraction",
    "non_sensitivity_delta_mean",
    "relative_input_stability",
    "covariate_complexity",
}

LOG_SCALED_HPARAMS = {
    "background_sample_size",
    "lime_num_samples",
    "causal_shap_coalitions",
    "ig_steps",
}

IDENTIFIER_COLUMNS = {
    "run_id",
    "client_id",
    "split",
    "selection_id",
    "shard_id",
    "explainer_name",
    "config_id",
    "instance_id",
    "dataset_index",
    "instance_index_within_job",
    "true_label",
    "prediction",
    "explained_class",
    "method_variant",
}


@dataclass(frozen=True)
class AggregatedCandidateSource:
    """One aggregation directory containing an instance_metrics table."""

    explainer_name: str
    config_id: str
    aggregation_dir: Path
    instance_metrics_path: Path
    aggregation_metadata_path: Path


def prepare_recommender_context(
    *,
    run_id: str,
    selection_id: str,
    explainers: str = "all",
    config_ids: str = "all",
    clients: str = "all",
    paths: ArtifactPaths | None = None,
) -> dict[str, Any]:
    """Build per-client context-feature files for recommender labeling/training."""

    _require_safe_segment(run_id, label="run_id")
    _require_safe_segment(selection_id, label="selection_id")
    artifact_paths = paths or ArtifactPaths()
    run_context = resolve_federated_run_context(paths=artifact_paths, run_id=run_id)
    requested_explainers = _split_selector(explainers)
    requested_configs = _split_selector(config_ids)
    requested_clients = _split_selector(clients)

    sources = _discover_aggregation_sources(
        run_artifact_dir=run_context.run_artifact_dir,
        selection_id=selection_id,
        explainer_filter=requested_explainers,
        config_filter=requested_configs,
    )
    if not sources:
        raise FileNotFoundError(
            "No aggregation instance_metrics.parquet files were found for "
            f"run_id={run_id!r}, selection={selection_id!r}, "
            f"explainers={explainers!r}, configs={config_ids!r}."
        )

    candidate_frame = _load_candidate_metrics(sources)
    if requested_clients is not None:
        candidate_frame = candidate_frame[candidate_frame["client_id"].isin(requested_clients)].copy()
    if candidate_frame.empty:
        raise ValueError("No candidate rows remain after applying client/explainer/config filters.")

    client_metadata = _build_client_dataset_metadata(run_context=run_context)
    client_metadata = _add_z_scores(client_metadata)
    if requested_clients is not None:
        missing_clients = sorted(set(requested_clients) - set(client_metadata["client_id"]))
        if missing_clients:
            raise ValueError(f"Requested clients were not found in the run metadata: {missing_clients}")

    encoded_frame = _encode_candidate_context(
        candidate_frame,
        client_metadata=client_metadata,
        run_metadata=run_context.metadata,
    )
    encoded_frame = _mark_pareto_candidates(encoded_frame)

    output_root = run_context.run_artifact_dir / "recommender_context" / selection_id
    client_summaries: list[dict[str, Any]] = []
    for client_id, client_frame in encoded_frame.groupby("client_id", sort=True):
        client_dir = run_context.run_artifact_dir / "clients" / client_id / "recommender_context" / selection_id
        all_path = client_dir / "all_candidate_context.parquet"
        pareto_path = client_dir / "candidate_context.parquet"
        pareto_json_path = client_dir / "pareto_front.json"
        metadata_path = client_dir / "preprocessing_metadata.json"

        pareto_frame = client_frame[client_frame["is_pareto_optimal"]].copy()
        _write_parquet_atomic(all_path, client_frame.reset_index(drop=True))
        _write_parquet_atomic(pareto_path, pareto_frame.reset_index(drop=True))
        pareto_payload = _build_pareto_json_payload(
            run_id=run_id,
            selection_id=selection_id,
            client_id=client_id,
            frame=pareto_frame,
        )
        _write_json_atomic(pareto_json_path, pareto_payload)

        client_summary = {
            "run_id": run_id,
            "selection_id": selection_id,
            "client_id": client_id,
            "candidate_count": int(len(client_frame)),
            "pareto_candidate_count": int(len(pareto_frame)),
            "instance_count": int(client_frame["dataset_index"].nunique()),
            "explainers": sorted(client_frame["explainer_name"].dropna().unique().tolist()),
            "configs": sorted(client_frame["config_id"].dropna().unique().tolist()),
            "artifacts": {
                "all_candidate_context": str(all_path),
                "candidate_context": str(pareto_path),
                "pareto_front": str(pareto_json_path),
                "preprocessing_metadata": str(metadata_path),
            },
            "generated_at": current_utc_timestamp(),
        }
        _write_json_atomic(metadata_path, client_summary)
        client_summaries.append(client_summary)

    manifest = {
        "status": "prepared",
        "run_id": run_id,
        "selection_id": selection_id,
        "generated_at": current_utc_timestamp(),
        "source_aggregation_count": len(sources),
        "candidate_count": int(len(encoded_frame)),
        "pareto_candidate_count": int(encoded_frame["is_pareto_optimal"].sum()),
        "client_count": len(client_summaries),
        "metric_features": sorted(
            column.removeprefix("metric_").removesuffix("_z")
            for column in encoded_frame.columns
            if column.startswith("metric_") and column.endswith("_z")
        ),
        "sources": [
            {
                "explainer_name": source.explainer_name,
                "config_id": source.config_id,
                "aggregation_dir": str(source.aggregation_dir),
                "instance_metrics_path": str(source.instance_metrics_path),
            }
            for source in sources
        ],
        "clients": client_summaries,
    }
    manifest_path = output_root / "preprocessing_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _discover_aggregation_sources(
    *,
    run_artifact_dir: Path,
    selection_id: str,
    explainer_filter: set[str] | None,
    config_filter: set[str] | None,
) -> list[AggregatedCandidateSource]:
    root = run_artifact_dir / "aggregations" / "explain_eval" / selection_id
    if not root.exists():
        return []
    sources: list[AggregatedCandidateSource] = []
    for explainer_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if explainer_filter is not None and explainer_dir.name not in explainer_filter:
            continue
        for config_dir in sorted(path for path in explainer_dir.iterdir() if path.is_dir()):
            if config_filter is not None and config_dir.name not in config_filter:
                continue
            instance_metrics_path = config_dir / "instance_metrics.parquet"
            if not instance_metrics_path.exists():
                continue
            sources.append(
                AggregatedCandidateSource(
                    explainer_name=explainer_dir.name,
                    config_id=config_dir.name,
                    aggregation_dir=config_dir,
                    instance_metrics_path=instance_metrics_path,
                    aggregation_metadata_path=config_dir / "aggregation_metadata.json",
                )
            )
    return sources


def _load_candidate_metrics(sources: Sequence[AggregatedCandidateSource]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in sources:
        frame = pd.read_parquet(source.instance_metrics_path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["explainer_name"] = frame.get("explainer_name", source.explainer_name)
        frame["config_id"] = frame.get("config_id", source.config_id)
        frame["method_variant"] = frame["config_id"]
        frame["source_instance_metrics_path"] = str(source.instance_metrics_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_client_dataset_metadata(*, run_context: Any) -> pd.DataFrame:
    training_metadata = json.loads(run_context.training_metadata_path.read_text(encoding="utf-8"))
    num_clients = int(training_metadata["num_clients"])
    client_datasets = load_client_datasets(run_context.partition_root, num_clients)
    rows: list[dict[str, Any]] = []
    for client in client_datasets:
        X_train = np.asarray(client.train.X, dtype=float)
        X_test = np.asarray(client.test.X, dtype=float)
        y_train = np.asarray(client.train.y)
        y_test = np.asarray(client.test.y)
        X_all = np.vstack([X_train, X_test]) if X_test.size else X_train
        y_all = np.concatenate([y_train, y_test]) if y_test.size else y_train
        feature_means = np.mean(X_train, axis=0) if X_train.size else np.asarray([], dtype=float)
        feature_stds = np.std(X_train, axis=0) if X_train.size else np.asarray([], dtype=float)
        rows.append(
            {
                "client_id": f"client_{int(client.client_id):03d}",
                "client_numeric_id": int(client.client_id),
                "client_train_size": int(X_train.shape[0]),
                "client_test_size": int(X_test.shape[0]),
                "client_total_size": int(X_all.shape[0]),
                "client_log_train_size": math.log1p(int(X_train.shape[0])),
                "client_log_total_size": math.log1p(int(X_all.shape[0])),
                "client_n_features": int(X_train.shape[1]) if X_train.ndim == 2 else 0,
                "client_class_entropy": _normalized_entropy(y_all),
                "client_train_positive_rate": _positive_rate(y_train),
                "client_test_positive_rate": _positive_rate(y_test),
                "client_feature_mean_abs_mean": _safe_float(np.mean(np.abs(feature_means))),
                "client_feature_mean_std": _safe_float(np.std(feature_means)),
                "client_feature_std_mean": _safe_float(np.mean(feature_stds)),
                "client_feature_std_std": _safe_float(np.std(feature_stds)),
                "client_feature_std_max": _safe_float(np.max(feature_stds) if feature_stds.size else 0.0),
                "client_feature_sparsity": _safe_float(np.mean(np.isclose(X_train, 0.0))) if X_train.size else 0.0,
                "client_mean_abs_correlation": _mean_abs_correlation(X_train),
            }
        )
    return pd.DataFrame(rows)


def _add_z_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    skip = {"client_id", "client_numeric_id"}
    for column in frame.columns:
        if column in skip or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].astype(float)
        std = float(values.std(ddof=0))
        result[f"{column}_z"] = 0.0 if std == 0.0 else (values - float(values.mean())) / std
    return result


def _encode_candidate_context(
    candidates: pd.DataFrame,
    *,
    client_metadata: pd.DataFrame,
    run_metadata: Mapping[str, Any],
) -> pd.DataFrame:
    metric_names = _metric_columns(candidates)
    configs = _resolved_configs(candidates)
    numeric_hp_stats, categorical_hp_values = _hyperparameter_feature_space(configs)
    explainer_metadata = _explainer_metadata()

    encoded_rows: list[dict[str, Any]] = []
    training_config = run_metadata.get("training_config") or {}
    dataset_name = str(run_metadata.get("dataset_name", training_config.get("dataset_name", "")))
    model_type = str(run_metadata.get("model_type", training_config.get("model_name", "")))
    for _, row in candidates.iterrows():
        explainer_name = str(row["explainer_name"])
        config_id = str(row["config_id"])
        client_id = str(row["client_id"])
        resolved_config = configs[(explainer_name, config_id)]
        encoded = {
            "run_id": str(row.get("run_id", "")),
            "dataset": dataset_name,
            "model": model_type,
            "client_id": client_id,
            "split": str(row.get("split", "")),
            "selection_id": str(row.get("selection_id", "")),
            "shard_id": str(row.get("shard_id", "")),
            "instance_index": _coerce_optional_int(row.get("dataset_index")),
            "dataset_index": _coerce_optional_int(row.get("dataset_index")),
            "instance_id": str(row.get("instance_id", "")),
            "true_label": row.get("true_label"),
            "prediction": row.get("prediction"),
            "explained_class": row.get("explained_class"),
            "method": explainer_name,
            "method_variant": config_id,
            "explainer_name": explainer_name,
            "config_id": config_id,
            "run_alpha": _coerce_optional_float((run_metadata.get("partition_reference") or {}).get("alpha")) or 0.0,
            "run_num_clients": _coerce_optional_float((run_metadata.get("partition_reference") or {}).get("num_clients")) or 0.0,
            "run_rounds": _coerce_optional_float(training_config.get("rounds")) or 0.0,
            "run_log_rounds": math.log1p(_coerce_optional_float(training_config.get("rounds")) or 0.0),
        }
        client_row = client_metadata.loc[client_metadata["client_id"] == client_id]
        if client_row.empty:
            raise ValueError(f"Missing client metadata for {client_id}.")
        encoded.update(
            {
                f"dataset_{key}": value
                for key, value in client_row.iloc[0].to_dict().items()
                if key != "client_id"
            }
        )
        encoded.update(_encode_explainer(explainer_name, explainer_metadata))
        encoded.update(
            _encode_hyperparameters(
                resolved_config,
                numeric_hp_stats=numeric_hp_stats,
                categorical_hp_values=categorical_hp_values,
            )
        )
        cleaned_metrics = _clean_oriented_metrics(row, metric_names)
        for metric_name, value in cleaned_metrics.items():
            encoded[f"metric_{metric_name}_oriented"] = value
        encoded["_oriented_metrics"] = cleaned_metrics
        encoded_rows.append(encoded)

    encoded_frame = pd.DataFrame(encoded_rows)
    return _add_metric_z_scores(encoded_frame, metric_names)


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    metric_names = []
    for column in frame.columns:
        if column in IDENTIFIER_COLUMNS or column.startswith("source_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            metric_names.append(str(column))
    return sorted(metric_names)


def _resolved_configs(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    configs: dict[tuple[str, str], dict[str, Any]] = {}
    for explainer_name, config_id in sorted(
        {(str(row.explainer_name), str(row.config_id)) for row in frame.itertuples()}
    ):
        configs[(explainer_name, config_id)] = resolve_explainer_config(explainer_name, config_id)
    return configs


def _hyperparameter_feature_space(
    configs: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    numeric_values: dict[str, list[float]] = {}
    categorical_values: dict[str, set[str]] = {}
    for config in configs.values():
        for key, value in config.items():
            numeric = _coerce_optional_float(value)
            if numeric is not None:
                transformed = math.log1p(numeric) if key in LOG_SCALED_HPARAMS and numeric >= 0 else numeric
                numeric_values.setdefault(key, []).append(float(transformed))
            elif isinstance(value, str):
                categorical_values.setdefault(key, set()).add(value)
    numeric_stats: dict[str, dict[str, float]] = {}
    for key, values in numeric_values.items():
        arr = np.asarray(values, dtype=float)
        numeric_stats[key] = {
            "mean": float(np.mean(arr)) if arr.size else 0.0,
            "std": float(np.std(arr)) if arr.size else 0.0,
            "log_scaled": 1.0 if key in LOG_SCALED_HPARAMS else 0.0,
        }
    categorical = {key: sorted(values) for key, values in categorical_values.items()}
    return numeric_stats, categorical


def _explainer_metadata() -> dict[str, dict[str, Any]]:
    names = DEFAULT_EXPLAINER_REGISTRY.list_keys()
    metadata: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        spec = DEFAULT_EXPLAINER_REGISTRY.get(name)
        family_token = str(spec.get("type") or name).lower()
        metadata[name] = {
            "explainer_id": index,
            "explainer_type": str(spec.get("type") or name),
            **_explainer_family_flags(name, family_token),
        }
    return metadata


def _encode_explainer(
    explainer_name: str,
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if explainer_name not in metadata:
        raise KeyError(f"Unknown explainer metadata for {explainer_name!r}.")
    row: dict[str, Any] = {}
    explainer_id = int(metadata[explainer_name]["explainer_id"])
    for name, item in metadata.items():
        row[f"explainer_id_oh_{int(item['explainer_id'])}"] = 1 if name == explainer_name else 0
    row["explainer_id"] = explainer_id
    for key, value in metadata[explainer_name].items():
        if key == "explainer_type":
            row["explainer_type"] = value
        elif key != "explainer_id":
            row[f"explainer_{key}"] = value
    return row


def _encode_hyperparameters(
    config: Mapping[str, Any],
    *,
    numeric_hp_stats: Mapping[str, Mapping[str, float]],
    categorical_hp_values: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, stats in sorted(numeric_hp_stats.items()):
        row[f"hp_is_applicable_{key}"] = 1 if key in config else 0
        numeric = _coerce_optional_float(config.get(key))
        if numeric is None:
            row[f"hp_{key}"] = 0.0
            continue
        value = math.log1p(numeric) if stats.get("log_scaled") and numeric >= 0 else numeric
        std = float(stats.get("std", 0.0))
        row[f"hp_{key}"] = 0.0 if std == 0.0 else (float(value) - float(stats.get("mean", 0.0))) / std
    for key, values in sorted(categorical_hp_values.items()):
        row[f"hp_is_applicable_{key}"] = 1 if key in config else 0
        current = str(config.get(key)) if key in config else ""
        for value in values:
            row[f"hp_{key}__{_slug(value)}"] = 1 if current == value else 0
    return row


def _clean_oriented_metrics(row: pd.Series, metric_names: Sequence[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_name in metric_names:
        if metric_name in IGNORED_PARETO_METRICS:
            continue
        value = _coerce_optional_float(row.get(metric_name))
        if value is None or not math.isfinite(value):
            continue
        if metric_name in LOWER_IS_BETTER_METRICS:
            value = -value
        metrics[metric_name] = value
    return metrics


def _add_metric_z_scores(frame: pd.DataFrame, metric_names: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    group_cols = ["client_id", "dataset_index"]
    for metric_name in metric_names:
        oriented_col = f"metric_{metric_name}_oriented"
        z_col = f"metric_{metric_name}_z"
        if oriented_col not in result.columns:
            continue
        result[z_col] = 0.0
        for _, indices in result.groupby(group_cols, dropna=False).groups.items():
            values = result.loc[indices, oriented_col].astype(float)
            std = float(values.std(ddof=0))
            result.loc[indices, z_col] = 0.0 if std == 0.0 else (values - float(values.mean())) / std
    return result.drop(columns=["_oriented_metrics"], errors="ignore")


def _mark_pareto_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    metric_cols = [
        column
        for column in result.columns
        if column.startswith("metric_") and column.endswith("_oriented")
    ]
    result["is_pareto_optimal"] = False
    result["candidate_index_within_instance"] = 0
    if not metric_cols:
        return result
    for _, group in result.groupby(["client_id", "dataset_index"], dropna=False, sort=False):
        indices = list(group.index)
        result.loc[indices, "candidate_index_within_instance"] = list(range(len(indices)))
        pareto_indices: list[int] = []
        for idx in indices:
            row_metrics = _row_metrics(result.loc[idx], metric_cols)
            if not row_metrics:
                continue
            dominated = False
            for other_idx in indices:
                if other_idx == idx:
                    continue
                other_metrics = _row_metrics(result.loc[other_idx], metric_cols)
                if _dominates(other_metrics, row_metrics):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(idx)
        result.loc[pareto_indices, "is_pareto_optimal"] = True
    return result


def _build_pareto_json_payload(
    *,
    run_id: str,
    selection_id: str,
    client_id: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    instances: list[dict[str, Any]] = []
    if not frame.empty:
        metric_cols = [
            column
            for column in frame.columns
            if column.startswith("metric_") and column.endswith("_oriented")
        ]
        for dataset_index, group in frame.groupby("dataset_index", sort=True):
            entries: list[dict[str, Any]] = []
            for _, row in group.iterrows():
                metrics = {
                    column.removeprefix("metric_").removesuffix("_oriented"): _coerce_optional_float(row[column])
                    for column in metric_cols
                    if _coerce_optional_float(row[column]) is not None
                }
                entries.append(
                    {
                        "method": row["explainer_name"],
                        "method_variant": row["config_id"],
                        "metrics": metrics,
                    }
                )
            instances.append(
                {
                    "dataset_index": int(dataset_index),
                    "instance_id": str(group.iloc[0].get("instance_id", "")),
                    "true_label": group.iloc[0].get("true_label"),
                    "predicted_label": group.iloc[0].get("prediction"),
                    "pareto_metrics": sorted(
                        column.removeprefix("metric_").removesuffix("_oriented")
                        for column in metric_cols
                    ),
                    "pareto_front": entries,
                }
            )
    return {
        "run_id": run_id,
        "selection_id": selection_id,
        "client_id": client_id,
        "n_instances": len(instances),
        "instances": instances,
        "generated_at": current_utc_timestamp(),
    }


def _row_metrics(row: pd.Series, metric_cols: Sequence[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for col in metric_cols:
        value = _coerce_optional_float(row.get(col))
        if value is not None and math.isfinite(value):
            metrics[col] = value
    return metrics


def _dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    metric_keys = sorted(set(left) | set(right))
    if not metric_keys:
        return False
    better_or_equal = True
    strictly_better = False
    for key in metric_keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            better_or_equal = False
            break
        if right_value is None:
            strictly_better = True
            continue
        if left_value < right_value:
            better_or_equal = False
            break
        if left_value > right_value:
            strictly_better = True
    return better_or_equal and strictly_better


def _normalized_entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    if counts.size <= 1:
        return 0.0
    return entropy / math.log(counts.size)


def _positive_rate(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values.astype(float)))


def _mean_abs_correlation(X: np.ndarray) -> float:
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return 0.0
    variances = np.var(X, axis=0)
    active = variances > 1e-12
    if int(np.sum(active)) < 2:
        return 0.0
    corr = np.corrcoef(X[:, active], rowvar=False)
    if corr.ndim != 2:
        return 0.0
    upper = corr[np.triu_indices_from(corr, k=1)]
    finite = upper[np.isfinite(upper)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite)))


def _explainer_family_flags(name: str, family_token: str) -> dict[str, int]:
    text = f"{name} {family_token}".lower()
    return {
        "is_additive_attribution": 1 if any(token in text for token in ("shap", "lime")) else 0,
        "is_gradient_based": 1 if any(token in text for token in ("gradient", "integrated")) else 0,
        "is_causal": 1 if "causal" in text else 0,
        "is_perturbation_based": 1 if any(token in text for token in ("lime", "kernel", "sampling")) else 0,
    }


def _split_selector(value: str) -> set[str] | None:
    text = str(value or "all").strip()
    if text.lower() == "all":
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def _require_safe_segment(value: str, *, label: str) -> None:
    segment = str(value)
    if not segment or segment in {".", ".."} or "/" in segment or "\\" in segment:
        raise ValueError(f"{label} must be a single non-empty path segment.")


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> int | None:
    numeric = _coerce_optional_float(value)
    return None if numeric is None else int(numeric)


def _safe_float(value: Any) -> float:
    numeric = _coerce_optional_float(value)
    if numeric is None or not math.isfinite(numeric):
        return 0.0
    return numeric


def _slug(value: Any) -> str:
    cleaned = []
    for char in str(value).lower().strip():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "value"


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Preparing recommender context requires pyarrow.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
