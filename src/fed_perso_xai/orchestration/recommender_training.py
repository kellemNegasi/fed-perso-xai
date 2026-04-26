"""Federated training and evaluation for explanation recommenders."""

from __future__ import annotations

import csv
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fed_perso_xai.fl.client import RecommenderClientData
from fed_perso_xai.fl.recommender_simulation import run_federated_recommender_training
from fed_perso_xai.orchestration.run_artifacts import resolve_federated_run_context
from fed_perso_xai.recommender import (
    PairwiseLogisticConfig,
    PairwiseLogisticRecommender,
    build_pairwise_recommender_data,
    evaluate_ranked_scores,
    infer_recommender_feature_columns,
    load_pairwise_logistic_recommender,
)
from fed_perso_xai.utils.config import ArtifactPaths, RecommenderFederatedTrainingConfig
from fed_perso_xai.utils.provenance import current_utc_timestamp


@dataclass(frozen=True)
class RecommenderTrainingArtifacts:
    """Stable filesystem pointers for one federated recommender run."""

    run_dir: Path
    model_artifact_path: Path
    model_metadata_path: Path
    feature_metadata_path: Path
    training_metadata_path: Path
    training_history_path: Path
    runtime_report_path: Path
    evaluation_summary_path: Path
    completion_marker_path: Path


def train_federated_recommender(
    config: RecommenderFederatedTrainingConfig,
    *,
    force: bool = False,
) -> tuple[RecommenderTrainingArtifacts, dict[str, Any]]:
    """Train a global pairwise recommender from client-local labeled preferences."""

    _require_safe_segment(config.run_id, label="run_id")
    _require_safe_segment(config.selection_id, label="selection_id")
    _require_safe_segment(config.persona, label="persona")
    run_context = resolve_federated_run_context(paths=config.paths, run_id=config.run_id)
    run_dir = _recommender_training_dir(run_context.run_artifact_dir, config.selection_id, config.persona)
    artifacts = _build_artifacts(run_dir)
    if artifacts.completion_marker_path.exists() and not force:
        metadata = json.loads(artifacts.training_metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "skipped_existing"
        metadata["skipped"] = True
        return artifacts, metadata
    if artifacts.completion_marker_path.exists():
        artifacts.completion_marker_path.unlink()

    loaded_clients = _load_client_recommender_inputs(
        run_artifact_dir=run_context.run_artifact_dir,
        selection_id=config.selection_id,
        persona=config.persona,
        clients=config.clients,
        context_filename=config.context_filename,
        label_filename=config.label_filename,
    )
    feature_columns = loaded_clients[0]["feature_columns"]
    model_config = PairwiseLogisticConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        l2_regularization=config.l2_regularization,
    )
    client_data = [
        RecommenderClientData(
            client_id=index,
            client_name=str(item["client_id"]),
            X_train=item["data"].X,
            y_train=item["data"].y,
            X_eval=item["data"].X,
            y_eval=item["data"].y,
        )
        for index, item in enumerate(loaded_clients)
    ]
    started_at = current_utc_timestamp()
    training_result = run_federated_recommender_training(
        client_datasets=client_data,
        config=config,
        model_config=model_config,
    )
    completed_at = current_utc_timestamp()

    model = PairwiseLogisticRecommender.from_config(
        n_features=len(feature_columns),
        config=model_config,
    )
    model.set_parameters(training_result.final_parameters)
    model.save(artifacts.model_artifact_path)
    _write_training_history_csv(artifacts.training_history_path, training_result.round_history)
    _write_json_atomic(artifacts.runtime_report_path, training_result.runtime_report)

    feature_metadata = {
        "artifact_type": "recommender_feature_metadata",
        "run_id": config.run_id,
        "selection_id": config.selection_id,
        "persona": config.persona,
        "feature_columns": list(feature_columns),
        "feature_count": int(len(feature_columns)),
        "context_filename": config.context_filename,
        "label_filename": config.label_filename,
    }
    _write_json_atomic(artifacts.feature_metadata_path, feature_metadata)

    evaluation = evaluate_recommender_model(
        run_id=config.run_id,
        selection_id=config.selection_id,
        persona=config.persona,
        model_path=artifacts.model_artifact_path,
        feature_columns=feature_columns,
        clients=config.clients,
        context_filename=config.context_filename,
        label_filename=config.label_filename,
        top_k=config.top_k,
        paths=config.paths,
        output_path=artifacts.evaluation_summary_path,
    )

    model_metadata = {
        "artifact_type": "federated_pairwise_recommender_model",
        "artifact_version": "pairwise_logistic_recommender_v1",
        "source_predictive_run_id": config.run_id,
        "selection_id": config.selection_id,
        "persona": config.persona,
        "model_type": "pairwise_logistic_recommender",
        "model_artifact_path": str(artifacts.model_artifact_path.relative_to(run_dir)),
        "feature_metadata_path": str(artifacts.feature_metadata_path.relative_to(run_dir)),
        "training_metadata_path": str(artifacts.training_metadata_path.relative_to(run_dir)),
        "evaluation_summary_path": str(artifacts.evaluation_summary_path.relative_to(run_dir)),
        "feature_count": int(len(feature_columns)),
        "parameter_count": int(len(training_result.final_parameters)),
        "model_config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "l2_regularization": config.l2_regularization,
        },
    }
    _write_json_atomic(artifacts.model_metadata_path, model_metadata)

    training_metadata = {
        "mode": "federated_recommender_training",
        "status": "completed",
        "source_predictive_run_id": config.run_id,
        "selection_id": config.selection_id,
        "persona": config.persona,
        "started_at": started_at,
        "completed_at": completed_at,
        "rounds_requested": int(config.rounds),
        "rounds_completed": int(len(training_result.round_history)),
        "simulation_backend_requested": config.simulation_backend,
        "simulation_backend_actual": training_result.actual_backend,
        "force_requested": bool(force),
        "client_count_loaded": int(len(loaded_clients)),
        "pair_count": int(sum(item["data"].augmented_pair_count for item in loaded_clients)),
        "raw_pair_count": int(sum(item["data"].pair_count for item in loaded_clients)),
        "candidate_count": int(sum(item["data"].candidate_count for item in loaded_clients)),
        "instance_count": int(sum(item["data"].instance_count for item in loaded_clients)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": list(feature_columns),
        "config": config.to_dict(),
        "model_artifact_path": str(artifacts.model_artifact_path.relative_to(run_dir)),
        "model_metadata_path": str(artifacts.model_metadata_path.relative_to(run_dir)),
        "feature_metadata_path": str(artifacts.feature_metadata_path.relative_to(run_dir)),
        "training_history_path": str(artifacts.training_history_path.relative_to(run_dir)),
        "runtime_report_path": str(artifacts.runtime_report_path.relative_to(run_dir)),
        "evaluation_summary_path": str(artifacts.evaluation_summary_path.relative_to(run_dir)),
        "clients": [
            {
                "client_id": str(item["client_id"]),
                "candidate_count": int(item["data"].candidate_count),
                "instance_count": int(item["data"].instance_count),
                "raw_pair_count": int(item["data"].pair_count),
                "augmented_pair_count": int(item["data"].augmented_pair_count),
                "context_path": str(item["context_path"]),
                "labels_path": str(item["labels_path"]),
            }
            for item in loaded_clients
        ],
        "evaluation": evaluation["aggregate"],
    }
    _write_json_atomic(artifacts.training_metadata_path, training_metadata)
    _write_json_atomic(
        artifacts.completion_marker_path,
        {
            "status": "completed",
            "completed_at": completed_at,
            "training_metadata_path": str(artifacts.training_metadata_path),
        },
    )
    return artifacts, training_metadata


def evaluate_recommender_model(
    *,
    run_id: str,
    selection_id: str,
    persona: str,
    model_path: Path | None = None,
    feature_columns: Sequence[str] | None = None,
    clients: str = "all",
    context_filename: str = "candidate_context.parquet",
    label_filename: str = "pairwise_labels.parquet",
    top_k: Iterable[int] = (1, 3, 5),
    paths: ArtifactPaths | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a global recommender against client-local labeled preferences."""

    _require_safe_segment(run_id, label="run_id")
    _require_safe_segment(selection_id, label="selection_id")
    _require_safe_segment(persona, label="persona")
    artifact_paths = paths or ArtifactPaths()
    run_context = resolve_federated_run_context(paths=artifact_paths, run_id=run_id)
    resolved_model_path = model_path or (
        _recommender_training_dir(run_context.run_artifact_dir, selection_id, persona)
        / "model"
        / "global_recommender.npz"
    )
    model = load_pairwise_logistic_recommender(resolved_model_path)
    if feature_columns is None:
        feature_metadata_path = resolved_model_path.parent.parent / "feature_metadata.json"
        if feature_metadata_path.exists():
            feature_metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
            feature_columns = tuple(str(column) for column in feature_metadata["feature_columns"])

    loaded_clients = _load_client_recommender_inputs(
        run_artifact_dir=run_context.run_artifact_dir,
        selection_id=selection_id,
        persona=persona,
        clients=clients,
        context_filename=context_filename,
        label_filename=label_filename,
        feature_columns=tuple(feature_columns) if feature_columns is not None else None,
    )
    if feature_columns is None:
        feature_columns = tuple(loaded_clients[0]["feature_columns"])
    client_metrics: list[dict[str, object]] = []
    for item in loaded_clients:
        candidates = item["candidates"]
        pair_labels = item["pair_labels"]
        scores = model.score_candidates(candidates, feature_columns)
        score_frame = scores.reset_index().rename(columns={"index": "method_variant"})
        variant_scores = (
            score_frame.groupby("method_variant", sort=True)["score"].mean().to_dict()
        )
        metrics = evaluate_ranked_scores(
            predicted_scores={str(key): float(value) for key, value in variant_scores.items()},
            pair_labels=pair_labels,
            top_k=top_k,
        )
        row: dict[str, object] = {
            "client_id": str(item["client_id"]),
            "candidate_count": int(len(candidates)),
            "pair_count": int(len(pair_labels)),
            "variant_count": int(metrics.get("variant_count", 0)),
            "pearson": float(metrics.get("pearson", 0.0)),
        }
        for key, value in metrics.items():
            if key.startswith("precision_at_"):
                row[key] = float(value)
        client_metrics.append(row)

    aggregate = _aggregate_client_metrics(client_metrics)
    payload = {
        "status": "evaluated",
        "run_id": run_id,
        "selection_id": selection_id,
        "persona": persona,
        "model_path": str(resolved_model_path),
        "feature_count": int(len(feature_columns)),
        "client_count": int(len(client_metrics)),
        "generated_at": current_utc_timestamp(),
        "aggregate": aggregate,
        "clients": client_metrics,
    }
    if output_path is not None:
        _write_json_atomic(output_path, payload)
    return payload


def _load_client_recommender_inputs(
    *,
    run_artifact_dir: Path,
    selection_id: str,
    persona: str,
    clients: str,
    context_filename: str,
    label_filename: str,
    feature_columns: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    requested_clients = _split_selector(clients)
    context_root = run_artifact_dir / "clients"
    client_dirs = [path for path in sorted(context_root.glob("client_*")) if path.is_dir()]
    if requested_clients is not None:
        client_dirs = [path for path in client_dirs if path.name in requested_clients]
        missing = sorted(requested_clients - {path.name for path in client_dirs})
        if missing:
            raise FileNotFoundError(f"Requested clients do not have run directories: {missing}")
    if not client_dirs:
        raise FileNotFoundError(f"No client directories found under {context_root}.")

    loaded: list[dict[str, Any]] = []
    resolved_features = feature_columns
    for client_dir in client_dirs:
        client_id = client_dir.name
        context_path = client_dir / "recommender_context" / selection_id / context_filename
        labels_path = client_dir / "recommender_labels" / selection_id / persona / label_filename
        if not context_path.exists() or not labels_path.exists():
            if requested_clients is not None:
                missing = context_path if not context_path.exists() else labels_path
                raise FileNotFoundError(f"Missing recommender input for {client_id}: {missing}")
            continue
        candidates = pd.read_parquet(context_path)
        pair_labels = pd.read_parquet(labels_path)
        if candidates.empty or pair_labels.empty:
            continue
        if resolved_features is None:
            resolved_features = infer_recommender_feature_columns(candidates)
        data = build_pairwise_recommender_data(
            candidates=candidates,
            pair_labels=pair_labels,
            feature_columns=resolved_features,
            augment_symmetric=True,
        )
        loaded.append(
            {
                "client_id": client_id,
                "context_path": context_path,
                "labels_path": labels_path,
                "candidates": candidates,
                "pair_labels": pair_labels,
                "data": data,
                "feature_columns": resolved_features,
            }
        )
    if not loaded:
        raise FileNotFoundError(
            "No labeled recommender inputs were found. Run prepare-recommender-context and "
            "label-recommender-context first."
        )
    return loaded


def _build_artifacts(run_dir: Path) -> RecommenderTrainingArtifacts:
    return RecommenderTrainingArtifacts(
        run_dir=run_dir,
        model_artifact_path=run_dir / "model" / "global_recommender.npz",
        model_metadata_path=run_dir / "model_metadata.json",
        feature_metadata_path=run_dir / "feature_metadata.json",
        training_metadata_path=run_dir / "training_metadata.json",
        training_history_path=run_dir / "training_history.csv",
        runtime_report_path=run_dir / "runtime_report.json",
        evaluation_summary_path=run_dir / "evaluation_summary.json",
        completion_marker_path=run_dir / "COMPLETED.json",
    )


def _recommender_training_dir(run_artifact_dir: Path, selection_id: str, persona: str) -> Path:
    return run_artifact_dir / "recommender_training" / selection_id / persona / "pairwise_logistic_fedavg"


def _write_training_history_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) or ["round"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_to_jsonable(value), sort_keys=True)
    return value


def _aggregate_client_metrics(clients: Sequence[Mapping[str, object]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    weights: dict[str, float] = {}
    for row in clients:
        weight = float(row.get("pair_count", 0) or 0)
        if weight <= 0:
            weight = 1.0
        for key, value in row.items():
            if key in {"client_id"} or key.endswith("count") or not isinstance(value, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value) * weight
            weights[key] = weights.get(key, 0.0) + weight
    return {key: sums[key] / weights[key] for key in sorted(sums) if weights.get(key, 0.0) > 0.0}


def _split_selector(value: str) -> set[str] | None:
    text = str(value or "all").strip()
    if text.lower() == "all":
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def _require_safe_segment(value: str, *, label: str) -> None:
    segment = str(value)
    if not segment or segment in {".", ".."} or "/" in segment or "\\" in segment:
        raise ValueError(f"{label} must be a single non-empty path segment.")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(_to_jsonable(dict(payload)), indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
