"""Standalone post-training explain+evaluate orchestration for federated runs.

Artifact model:
- one client-level metadata file per run/client
- one shared selection manifest per (run_id, client_id, split, max_instances, random_state)
- one shard manifest per (selection_id, shard_id)
- one explain/eval job per (selection_id, shard_id, explainer_name, config_id)
"""

from __future__ import annotations

import json
import os
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fed_perso_xai.data.serialization import load_client_datasets
from fed_perso_xai.evaluators import (
    DEFAULT_METRIC_REGISTRY,
    evaluate_metrics_for_method,
    extract_metric_parameters,
    make_metric,
    metric_capabilities,
)
from fed_perso_xai.explainers import resolve_explainer_config
from fed_perso_xai.fl.client import ClientData
from fed_perso_xai.models import load_global_model
from fed_perso_xai.orchestration.explanations import (
    LocalExplanationDataset,
    instantiate_explainer,
    load_feature_names_from_metadata,
    to_serializable,
)
from fed_perso_xai.orchestration.run_artifacts import FederatedRunContext, resolve_federated_run_context
from fed_perso_xai.utils.config import ArtifactPaths
from fed_perso_xai.utils.paths import (
    federated_client_metadata_path,
    federated_detailed_explanations_dir,
    federated_job_status_dir,
    federated_metrics_results_dir,
    federated_selection_id,
    federated_selection_metadata_path,
    federated_shard_metadata_path,
)
from fed_perso_xai.utils.provenance import current_utc_timestamp

DEFAULT_SHARD_SIZE = 1024


@dataclass(frozen=True)
class ExplainEvalJobArtifacts:
    """Filesystem pointers for one atomic explain+evaluate job."""

    run_artifact_dir: Path
    explanations_path: Path
    metrics_path: Path
    selection_metadata_path: Path
    shard_metadata_path: Path
    client_metadata_path: Path
    job_metadata_path: Path
    done_marker_path: Path


def run_explain_eval_job(
    run_id: str,
    client_id: str | int,
    split: str = "test",
    shard_id: str = "shard_000",
    explainer_name: str = "lime",
    config_id: str = "lime__kernel-1.5__samples-50",
    max_instances: int = 50,
    random_state: int = 42,
    *,
    force: bool = False,
    paths: ArtifactPaths | None = None,
    metric_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run one atomic post-training explain+evaluate job for a client shard."""

    artifact_paths = paths or ArtifactPaths()
    run_context = resolve_federated_run_context(paths=artifact_paths, run_id=run_id)
    normalized_client_id, client_numeric_id = _normalize_client_id(client_id)
    split_name = _normalize_split_name(split)
    selection_id = federated_selection_id(
        split=split_name,
        max_instances=max_instances,
        random_state=random_state,
    )
    artifacts = _build_job_artifacts(
        run_context=run_context,
        client_id=normalized_client_id,
        selection_id=selection_id,
        shard_id=shard_id,
        explainer_name=explainer_name,
        config_id=config_id,
    )

    if _job_is_complete(artifacts) and not force:
        payload = json.loads(artifacts.job_metadata_path.read_text(encoding="utf-8"))
        payload["status"] = "skipped_existing"
        payload["skipped"] = True
        return payload

    _remove_if_exists(artifacts.done_marker_path)

    run_model = load_global_model(run_context.run_artifact_dir)
    feature_names = load_feature_names_from_metadata(run_context.feature_metadata_path)
    client_data = _load_client_data(
        run_context=run_context,
        client_numeric_id=client_numeric_id,
    )
    X_split, y_split, row_ids_split = client_data.get_split(split_name)
    selected_view = _select_split_instances(
        X=X_split,
        y=y_split,
        row_ids=row_ids_split,
        max_instances=max_instances,
        random_state=random_state,
        run_id=run_id,
        client_id=normalized_client_id,
        split=split_name,
    )
    shard_view = _resolve_selected_shard(
        X=selected_view["X"],
        y=selected_view["y"],
        row_ids=selected_view["row_ids"],
        dataset_indices=selected_view["dataset_indices"],
        shard_id=shard_id,
    )
    X_job = np.asarray(shard_view["X"], dtype=np.float64)
    y_job = np.asarray(shard_view["y"], dtype=np.int64)
    row_ids_job = np.asarray(shard_view["row_ids"], dtype=str)
    dataset_indices_job = np.asarray(shard_view["dataset_indices"], dtype=np.int64)

    resolved_config = resolve_explainer_config(explainer_name, config_id)
    resolved_config = _pin_explainer_to_assigned_rows(
        resolved_config,
        assigned_instance_count=int(len(X_job)),
        random_state=random_state,
    )
    dataset = LocalExplanationDataset(
        X_train=np.asarray(client_data.X_train, dtype=np.float64),
        y_train=np.asarray(client_data.y_train, dtype=np.int64),
        feature_names=feature_names,
    )
    explainer = instantiate_explainer(
        explainer_name,
        model=run_model.model,
        dataset=dataset,
        params_override=resolved_config,
    )
    explainer_results = explainer.explain_dataset(X_job, y_job)

    explainer_sample_indices = explainer.sample_indices()
    if explainer_sample_indices is not None:
        row_ids_job = row_ids_job[explainer_sample_indices]
        dataset_indices_job = dataset_indices_job[explainer_sample_indices]

    explanation_rows = _build_explanation_rows(
        run_id=run_id,
        client_id=normalized_client_id,
        split=split_name,
        selection_id=selection_id,
        shard_id=shard_id,
        explainer_name=explainer_name,
        config_id=config_id,
        resolved_config=resolved_config,
        row_ids=row_ids_job,
        dataset_indices=dataset_indices_job,
        explanations=explainer_results["explanations"],
    )

    metrics_payload = _evaluate_job_metrics(
        metric_names=metric_names,
        model=run_model.model,
        dataset=dataset,
        explainer=explainer,
        explainer_name=explainer_name,
        config_id=config_id,
        run_id=run_id,
        client_id=normalized_client_id,
        split=split_name,
        selection_id=selection_id,
        shard_id=shard_id,
        resolved_config=resolved_config,
        row_ids=row_ids_job,
        dataset_indices=dataset_indices_job,
        explanations=explainer_results["explanations"],
        shard_view=shard_view,
        selected_view=selected_view,
        run_context=run_context,
    )

    selection_metadata = {
        "run_id": run_id,
        "client_id": normalized_client_id,
        "split": split_name,
        "selection_id": selection_id,
        "original_split_size": int(len(X_split)),
        "selection_subset_size": int(len(selected_view["X"])),
        "selection_dataset_indices": selected_view["dataset_indices"].tolist(),
        "selection_row_ids": selected_view["row_ids"].tolist(),
        "random_state": int(random_state),
        "max_instances": int(max_instances),
        "selection_seed": int(selected_view["selection_seed"]),
        "selection_strategy": "stable_random_without_replacement",
        "generated_at": current_utc_timestamp(),
    }
    shard_metadata = {
        "run_id": run_id,
        "client_id": normalized_client_id,
        "split": split_name,
        "selection_id": selection_id,
        "shard_id": shard_id,
        "rows_per_shard": DEFAULT_SHARD_SIZE,
        "shard_index_within_selection": int(shard_view["shard_index"]),
        "shard_count_for_selection": int(shard_view["shard_count"]),
        "selected_position_start": int(shard_view["selection_position_start"]),
        "selected_position_end_exclusive": int(shard_view["selection_position_end_exclusive"]),
        "shard_size": int(len(shard_view["X"])),
        "generated_at": current_utc_timestamp(),
    }
    client_metadata = {
        "run_id": run_id,
        "client_id": normalized_client_id,
        "client_numeric_id": client_numeric_id,
        "partition_root": str(run_context.partition_root),
        "split_availability": ["train", "test"],
        "train_size": int(client_data.X_train.shape[0]),
        "test_size": int(client_data.X_test.shape[0]),
        "generated_at": current_utc_timestamp(),
    }
    job_metadata = {
        "status": "completed",
        "run_id": run_id,
        "client_id": normalized_client_id,
        "client_numeric_id": client_numeric_id,
        "split": split_name,
        "selection_id": selection_id,
        "shard_id": shard_id,
        "explainer_name": explainer_name,
        "config_id": config_id,
        "selected_config": resolved_config,
        "metric_set": metrics_payload["metric_names"],
        "selection_subset_size": int(len(selected_view["X"])),
        "selected_dataset_indices": dataset_indices_job.tolist(),
        "selected_row_ids": row_ids_job.tolist(),
        "max_instances": int(max_instances),
        "random_state": int(random_state),
        "artifacts": {
            "detailed_explanations_path": str(artifacts.explanations_path),
            "metrics_results_path": str(artifacts.metrics_path),
            "selection_metadata_path": str(artifacts.selection_metadata_path),
            "shard_metadata_path": str(artifacts.shard_metadata_path),
            "client_metadata_path": str(artifacts.client_metadata_path),
            "job_metadata_path": str(artifacts.job_metadata_path),
            "done_marker_path": str(artifacts.done_marker_path),
        },
        "generated_at": current_utc_timestamp(),
    }

    _write_json_atomic(artifacts.client_metadata_path, client_metadata)
    _write_json_if_missing_atomic(artifacts.selection_metadata_path, selection_metadata)
    _write_json_atomic(artifacts.shard_metadata_path, shard_metadata)
    _write_parquet_atomic(artifacts.explanations_path, explanation_rows)
    _write_json_atomic(artifacts.metrics_path, metrics_payload)
    _write_json_atomic(artifacts.job_metadata_path, job_metadata)
    _write_json_atomic(
        artifacts.done_marker_path,
        {
            "status": "completed",
            "run_id": run_id,
            "client_id": normalized_client_id,
            "split": split_name,
            "selection_id": selection_id,
            "shard_id": shard_id,
            "explainer_name": explainer_name,
            "config_id": config_id,
            "completed_at": current_utc_timestamp(),
        },
    )

    return job_metadata


def _load_client_data(
    *,
    run_context: FederatedRunContext,
    client_numeric_id: int,
) -> ClientData:
    training_metadata = json.loads(run_context.training_metadata_path.read_text(encoding="utf-8"))
    client_datasets = load_client_datasets(
        run_context.partition_root,
        int(training_metadata["num_clients"]),
    )
    try:
        loaded = next(item for item in client_datasets if item.client_id == client_numeric_id)
    except StopIteration as exc:
        raise ValueError(
            f"Client '{client_numeric_id}' was not found under '{run_context.partition_root}'."
        ) from exc
    return ClientData(
        client_id=loaded.client_id,
        X_train=loaded.train.X,
        y_train=loaded.train.y,
        row_ids_train=loaded.train.row_ids,
        X_test=loaded.test.X,
        y_test=loaded.test.y,
        row_ids_test=loaded.test.row_ids,
    )


def _select_split_instances(
    *,
    X: np.ndarray,
    y: np.ndarray,
    row_ids: np.ndarray,
    max_instances: int,
    random_state: int,
    run_id: str,
    client_id: str,
    split: str,
) -> dict[str, Any]:
    if max_instances <= 0:
        raise ValueError("max_instances must be positive.")
    total = int(len(X))
    dataset_indices = np.arange(total, dtype=np.int64)
    selection_id = federated_selection_id(
        split=split,
        max_instances=max_instances,
        random_state=random_state,
    )
    selection_seed = _stable_selection_seed(
        run_id=run_id,
        client_id=client_id,
        split=split,
        random_state=random_state,
        max_instances=max_instances,
    )
    if total <= max_instances:
        selected_positions = np.arange(total, dtype=np.int64)
    else:
        rng = np.random.default_rng(selection_seed)
        selected_positions = np.sort(rng.choice(total, size=max_instances, replace=False).astype(np.int64))
    return {
        "selection_id": selection_id,
        "max_instances": int(max_instances),
        "random_state": int(random_state),
        "selection_seed": selection_seed,
        "selected_positions": selected_positions,
        "dataset_indices": dataset_indices[selected_positions],
        "X": np.asarray(X[selected_positions], dtype=np.float64),
        "y": np.asarray(y[selected_positions], dtype=np.int64),
        "row_ids": np.asarray(row_ids[selected_positions], dtype=str),
    }


def _resolve_selected_shard(
    *,
    X: np.ndarray,
    y: np.ndarray,
    row_ids: np.ndarray,
    dataset_indices: np.ndarray,
    shard_id: str,
) -> dict[str, Any]:
    shard_index = _parse_shard_id(shard_id)
    total = int(len(X))
    start = shard_index * DEFAULT_SHARD_SIZE
    end = min(total, start + DEFAULT_SHARD_SIZE)
    if start >= total:
        raise ValueError(
            f"Shard '{shard_id}' is out of range for selected subset size {total} with shard size {DEFAULT_SHARD_SIZE}."
        )
    shard_count = int((total + DEFAULT_SHARD_SIZE - 1) // DEFAULT_SHARD_SIZE)
    return {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selection_position_start": start,
        "selection_position_end_exclusive": end,
        "dataset_indices": np.asarray(dataset_indices[start:end], dtype=np.int64),
        "X": np.asarray(X[start:end], dtype=np.float64),
        "y": np.asarray(y[start:end], dtype=np.int64),
        "row_ids": np.asarray(row_ids[start:end], dtype=str),
    }


def _build_explanation_rows(
    *,
    run_id: str,
    client_id: str,
    split: str,
    selection_id: str,
    shard_id: str,
    explainer_name: str,
    config_id: str,
    resolved_config: dict[str, Any],
    row_ids: np.ndarray,
    dataset_indices: np.ndarray,
    explanations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job_index, explanation in enumerate(explanations):
        metadata = dict(explanation.get("metadata") or {})
        method_variant = str(
            metadata.get("method_variant")
            or explanation.get("method")
            or explainer_name
        )
        rows.append(
            {
                "run_id": run_id,
                "client_id": client_id,
                "split": split,
                "selection_id": selection_id,
                "shard_id": shard_id,
                "explainer_name": explainer_name,
                "config_id": config_id,
                "method_variant": method_variant,
                "instance_id": str(row_ids[job_index]),
                "dataset_index": int(dataset_indices[job_index]),
                "instance_index_within_job": int(job_index),
                "true_label": _json_scalar(metadata.get("true_label")),
                "prediction": _json_scalar(explanation.get("prediction")),
                "explained_class": _json_scalar(metadata.get("explained_class")),
                "prediction_proba_json": json.dumps(to_serializable(explanation.get("prediction_proba")), sort_keys=True),
                "attributions_json": json.dumps(to_serializable(explanation.get("attributions")), sort_keys=True),
                "instance_json": json.dumps(to_serializable(explanation.get("instance")), sort_keys=True),
                "feature_names_json": json.dumps(to_serializable(explanation.get("feature_names")), sort_keys=True),
                "explanation_metadata_json": json.dumps(to_serializable(metadata), sort_keys=True),
                "explainer_config_json": json.dumps(to_serializable(resolved_config), sort_keys=True),
                "generation_time_seconds": float(explanation.get("generation_time", 0.0) or 0.0),
                "generated_at": current_utc_timestamp(),
            }
        )
    return rows


def _evaluate_job_metrics(
    *,
    metric_names: list[str] | None,
    model: Any,
    dataset: Any,
    explainer: Any,
    explainer_name: str,
    config_id: str,
    run_id: str,
    client_id: str,
    split: str,
    selection_id: str,
    shard_id: str,
    resolved_config: dict[str, Any],
    row_ids: np.ndarray,
    dataset_indices: np.ndarray,
    explanations: list[dict[str, Any]],
    shard_view: dict[str, Any],
    selected_view: dict[str, Any],
    run_context: FederatedRunContext,
) -> dict[str, Any]:
    active_metric_names = metric_names or [
        name for name in DEFAULT_METRIC_REGISTRY.list_keys()
        if DEFAULT_METRIC_REGISTRY.is_available(name)
    ]
    metric_objs = {name: make_metric(name) for name in active_metric_names}
    metric_caps = {name: metric_capabilities(metric) for name, metric in metric_objs.items()}

    explainer_results = {
        "method": explainer_name,
        "client_id": client_id,
        "split_name": split,
        "row_ids": row_ids.tolist(),
        "explanations": explanations,
    }
    dataset_mapping = {
        int(dataset_idx): (int(job_index), explanations[job_index])
        for job_index, dataset_idx in enumerate(dataset_indices)
    }
    execution = evaluate_metrics_for_method(
        metric_objs=metric_objs,
        metric_caps=metric_caps,
        explainer=explainer,
        expl_results=explainer_results,
        dataset_mapping=dataset_mapping,
        model=model,
        dataset=dataset,
        method_label=explainer_name,
        log_progress=False,
    )

    instance_results: list[dict[str, Any]] = []
    for job_index, explanation in enumerate(explanations):
        dataset_index = int(dataset_indices[job_index])
        metric_values = execution.instance_metrics.get(dataset_index, {}).get(job_index, {})
        metadata = dict(explanation.get("metadata") or {})
        instance_results.append(
            {
                "instance_id": str(row_ids[job_index]),
                "dataset_index": dataset_index,
                "instance_index_within_job": int(job_index),
                "true_label": to_serializable(metadata.get("true_label")),
                "prediction": to_serializable(explanation.get("prediction")),
                "explained_class": to_serializable(metadata.get("explained_class")),
                "method_variant": str(metadata.get("method_variant") or explanation.get("method") or explainer_name),
                "metrics": to_serializable(metric_values),
            }
        )

    metric_metadata = {
        metric_name: {
            "params": extract_metric_parameters(metric_obj),
            "capabilities": metric_caps[metric_name],
        }
        for metric_name, metric_obj in metric_objs.items()
    }
    return {
        "run_id": run_id,
        "client_id": client_id,
        "split": split,
        "selection_id": selection_id,
        "shard_id": shard_id,
        "explainer_name": explainer_name,
        "config_id": config_id,
        "generated_at": current_utc_timestamp(),
        "model_traceability": {
            "model_type": run_context.metadata["model_type"],
            "run_metadata_path": str(run_context.run_metadata_path),
            "training_metadata_path": str(run_context.training_metadata_path),
            "model_artifact_path": str(run_context.model_artifact_path),
            "training_config_sha256": run_context.metadata["training_config_sha256"],
        },
        "selected_config": resolved_config,
        "metric_names": active_metric_names,
        "metric_metadata": metric_metadata,
        "selection_metadata": {
            "selection_id": selected_view["selection_id"],
            "selection_subset_size": int(len(selected_view["X"])),
            "selection_dataset_indices": selected_view["dataset_indices"].tolist(),
            "random_state": int(selected_view["random_state"]),
            "max_instances": int(selected_view["max_instances"]),
            "selection_seed": int(selected_view["selection_seed"]),
        },
        "shard_metadata": {
            "rows_per_shard": DEFAULT_SHARD_SIZE,
            "selection_id": selected_view["selection_id"],
            "shard_index_within_selection": int(shard_view["shard_index"]),
            "shard_count_for_selection": int(shard_view["shard_count"]),
            "selected_position_start": int(shard_view["selection_position_start"]),
            "selected_position_end_exclusive": int(shard_view["selection_position_end_exclusive"]),
            "shard_size": int(len(shard_view["X"])),
            "selected_instance_count": int(len(explanations)),
        },
        "per_instance_results": instance_results,
        "batch_metrics": to_serializable(execution.batch_metrics),
        "batch_metrics_by_variant": {
            explainer_name: to_serializable(execution.batch_metrics),
        },
    }


def _build_job_artifacts(
    *,
    run_context: FederatedRunContext,
    client_id: str,
    selection_id: str,
    shard_id: str,
    explainer_name: str,
    config_id: str,
) -> ExplainEvalJobArtifacts:
    explanations_dir = federated_detailed_explanations_dir(
        run_context.run_artifact_dir,
        client_id,
        selection_id,
        shard_id,
        explainer_name,
    )
    metrics_dir = federated_metrics_results_dir(
        run_context.run_artifact_dir,
        client_id,
        selection_id,
        shard_id,
        explainer_name,
    )
    status_dir = federated_job_status_dir(
        run_context.run_artifact_dir,
        client_id,
        selection_id,
        shard_id,
    )
    return ExplainEvalJobArtifacts(
        run_artifact_dir=run_context.run_artifact_dir,
        explanations_path=explanations_dir / f"{config_id}.parquet",
        metrics_path=metrics_dir / f"{config_id}.json",
        selection_metadata_path=federated_selection_metadata_path(
            run_context.run_artifact_dir,
            client_id,
            selection_id,
        ),
        shard_metadata_path=federated_shard_metadata_path(
            run_context.run_artifact_dir,
            client_id,
            selection_id,
            shard_id,
        ),
        client_metadata_path=federated_client_metadata_path(
            run_context.run_artifact_dir,
            client_id,
        ),
        job_metadata_path=status_dir / f"{config_id}.json",
        done_marker_path=status_dir / f"{config_id}.done",
    )


def _job_is_complete(artifacts: ExplainEvalJobArtifacts) -> bool:
    return all(
        path.exists()
        for path in (
            artifacts.explanations_path,
            artifacts.metrics_path,
            artifacts.selection_metadata_path,
            artifacts.shard_metadata_path,
            artifacts.client_metadata_path,
            artifacts.job_metadata_path,
            artifacts.done_marker_path,
        )
    )


def _normalize_client_id(client_id: str | int) -> tuple[str, int]:
    if isinstance(client_id, int):
        return f"client_{client_id:03d}", client_id
    text = str(client_id).strip()
    if text.startswith("client_"):
        numeric = int(text.split("_", 1)[1])
        return f"client_{numeric:03d}", numeric
    numeric = int(text)
    return f"client_{numeric:03d}", numeric


def _normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")
    return normalized


def _parse_shard_id(shard_id: str) -> int:
    if not shard_id.startswith("shard_"):
        raise ValueError(f"Invalid shard_id '{shard_id}'. Expected format 'shard_000'.")
    shard_index = int(shard_id.split("_", 1)[1])
    if shard_index < 0:
        raise ValueError(f"Invalid shard_id '{shard_id}'. Shard index must be non-negative.")
    return shard_index


def _stable_selection_seed(
    run_id: str,
    client_id: str,
    split: str,
    random_state: int,
    max_instances: int,
    ) -> int:
    seed_payload = json.dumps(
        {
            "run_id": run_id,
            "client_id": client_id,
            "split": split,
            "random_state": int(random_state),
            "max_instances": int(max_instances),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(seed_payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _pin_explainer_to_assigned_rows(
    resolved_config: dict[str, Any],
    *,
    assigned_instance_count: int,
    random_state: int,
) -> dict[str, Any]:
    pinned = dict(resolved_config)
    pinned["max_instances"] = int(assigned_instance_count)
    pinned["method_max_instances"] = int(assigned_instance_count)
    pinned["sampling_strategy"] = "sequential"
    pinned["random_state"] = int(random_state)
    return pinned


def _json_scalar(value: Any) -> str:
    return json.dumps(to_serializable(value), sort_keys=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_json_if_missing_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(to_serializable(payload), indent=2)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        # Another concurrent job won the manifest creation race.
        return


def _write_parquet_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Writing explanation artifacts requires 'pyarrow'. Install the project dependencies "
            "with Parquet support before running explain+evaluate jobs."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    schema = pa.schema(
        [
            ("run_id", pa.string()),
            ("client_id", pa.string()),
            ("split", pa.string()),
            ("selection_id", pa.string()),
            ("shard_id", pa.string()),
            ("explainer_name", pa.string()),
            ("config_id", pa.string()),
            ("method_variant", pa.string()),
            ("instance_id", pa.string()),
            ("dataset_index", pa.int64()),
            ("instance_index_within_job", pa.int64()),
            ("true_label", pa.string()),
            ("prediction", pa.string()),
            ("explained_class", pa.string()),
            ("prediction_proba_json", pa.string()),
            ("attributions_json", pa.string()),
            ("instance_json", pa.string()),
            ("feature_names_json", pa.string()),
            ("explanation_metadata_json", pa.string()),
            ("explainer_config_json", pa.string()),
            ("generation_time_seconds", pa.float64()),
            ("generated_at", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, tmp_path)
    tmp_path.replace(path)


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
