"""Federated training and evaluation for explanation recommenders."""

from __future__ import annotations

import csv
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from fed_perso_xai.fl.client import RecommenderClientData
from fed_perso_xai.fl.recommender_simulation import (
    RecommenderSimulationArtifacts,
    run_federated_recommender_training,
)
from fed_perso_xai.orchestration.run_artifacts import resolve_federated_run_context
from fed_perso_xai.recommender import (
    DEFAULT_RECOMMENDER_TYPE,
    PairwiseLogisticConfig,
    build_pairwise_recommender_data,
    create_recommender,
    evaluate_grouped_ranked_scores,
    infer_recommender_feature_columns,
    load_recommender,
    recommender_artifact_model_type,
)
from fed_perso_xai.recommender.evaluation import is_recommender_metric_key
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
    cluster_manifest_path: Path
    cluster_rounds_dir: Path
    cluster_models_dir: Path
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
    _require_safe_segment(config.context_filename, label="context_filename")
    _require_safe_segment(config.label_filename, label="label_filename")
    run_context = resolve_federated_run_context(paths=config.paths, run_id=config.run_id)
    run_dir = _recommender_training_dir(
        run_context.run_artifact_dir,
        config.selection_id,
        config.persona,
        recommender_type=config.recommender_type,
    )
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
        split_name="train",
    )
    feature_columns = loaded_clients[0]["feature_columns"]
    try:
        loaded_eval_clients = _load_client_recommender_inputs(
            run_artifact_dir=run_context.run_artifact_dir,
            selection_id=config.selection_id,
            persona=config.persona,
            clients=config.clients,
            context_filename=config.context_filename,
            label_filename=config.label_filename,
            feature_columns=feature_columns,
            split_name="test",
        )
    except FileNotFoundError:
        loaded_eval_clients = []
    eval_lookup = {str(item["client_id"]): item for item in loaded_eval_clients}
    missing_eval_clients = sorted(
        str(item["client_id"])
        for item in loaded_clients
        if str(item["client_id"]) not in eval_lookup
    )
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
            X_eval=(
                eval_lookup[str(item["client_id"])]["data"].X
                if str(item["client_id"]) in eval_lookup
                else np.empty((0, len(feature_columns)), dtype=np.float64)
            ),
            y_eval=(
                eval_lookup[str(item["client_id"])]["data"].y
                if str(item["client_id"]) in eval_lookup
                else np.empty((0,), dtype=np.int64)
            ),
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

    model = create_recommender(
        recommender_type=config.recommender_type,
        n_features=len(feature_columns),
        config=model_config,
    )
    model.set_parameters(training_result.final_parameters)
    model.save(artifacts.model_artifact_path)
    cluster_manifest = (
        _persist_clustered_training_artifacts(
            artifacts=artifacts,
            run_dir=run_dir,
            training_result=training_result,
            config=config,
            feature_columns=feature_columns,
            model_config=model_config,
        )
        if training_result.clustered
        else None
    )
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

    if loaded_eval_clients:
        if training_result.clustered:
            if cluster_manifest is None:
                raise RuntimeError("Clustered recommender training completed without a cluster manifest.")
            evaluation = _evaluate_clustered_recommender_models(
                loaded_clients=loaded_eval_clients,
                feature_columns=feature_columns,
                cluster_assignments=training_result.final_cluster_assignments,
                cluster_model_paths={
                    cluster_id: run_dir / relative_path
                    for cluster_id, relative_path in cluster_manifest[
                        "final_cluster_model_checkpoint_paths"
                    ].items()
                },
                run_id=config.run_id,
                selection_id=config.selection_id,
                persona=config.persona,
                recommender_type=config.recommender_type,
                top_k=config.top_k,
                output_path=artifacts.evaluation_summary_path,
            )
        else:
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
                recommender_type=config.recommender_type,
            )
    else:
        evaluation = {
            "status": "skipped_no_test_pairs",
            "run_id": config.run_id,
            "selection_id": config.selection_id,
            "persona": config.persona,
            "model_path": str(artifacts.model_artifact_path),
            "feature_count": int(len(feature_columns)),
            "client_count": 0,
            "generated_at": current_utc_timestamp(),
            "aggregate": {},
            "clients": [],
            "reason": "No held-out recommender evaluation pairs were available for any client.",
        }
        _write_json_atomic(artifacts.evaluation_summary_path, evaluation)

    model_metadata = {
        "artifact_type": "federated_pairwise_recommender_model",
        "artifact_version": "federated_pairwise_recommender_v2",
        "source_predictive_run_id": config.run_id,
        "selection_id": config.selection_id,
        "persona": config.persona,
        "recommender_type": config.recommender_type,
        "model_type": recommender_artifact_model_type(config.recommender_type),
        "model_artifact_path": str(artifacts.model_artifact_path.relative_to(run_dir)),
        "feature_metadata_path": str(artifacts.feature_metadata_path.relative_to(run_dir)),
        "training_metadata_path": str(artifacts.training_metadata_path.relative_to(run_dir)),
        "evaluation_summary_path": str(artifacts.evaluation_summary_path.relative_to(run_dir)),
        "feature_count": int(len(feature_columns)),
        "parameter_count": int(len(training_result.final_parameters)),
        "clustered": bool(training_result.clustered),
        "cluster_manifest_path": (
            str(artifacts.cluster_manifest_path.relative_to(run_dir))
            if training_result.clustered
            else None
        ),
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
        "recommender_type": config.recommender_type,
        "clustered": bool(training_result.clustered),
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
        "eval_pair_count": int(sum(item["data"].augmented_pair_count for item in loaded_eval_clients)),
        "eval_raw_pair_count": int(sum(item["data"].pair_count for item in loaded_eval_clients)),
        "eval_candidate_count": int(sum(item["data"].candidate_count for item in loaded_eval_clients)),
        "eval_instance_count": int(sum(item["data"].instance_count for item in loaded_eval_clients)),
        "clients_without_eval": missing_eval_clients,
        "feature_count": int(len(feature_columns)),
        "feature_columns": list(feature_columns),
        "config": config.to_dict(),
        "model_artifact_path": str(artifacts.model_artifact_path.relative_to(run_dir)),
        "model_metadata_path": str(artifacts.model_metadata_path.relative_to(run_dir)),
        "feature_metadata_path": str(artifacts.feature_metadata_path.relative_to(run_dir)),
        "training_history_path": str(artifacts.training_history_path.relative_to(run_dir)),
        "runtime_report_path": str(artifacts.runtime_report_path.relative_to(run_dir)),
        "evaluation_summary_path": str(artifacts.evaluation_summary_path.relative_to(run_dir)),
        "cluster_manifest_path": (
            str(artifacts.cluster_manifest_path.relative_to(run_dir))
            if training_result.clustered
            else None
        ),
        "final_cluster_assignments": (
            dict(training_result.final_cluster_assignments) if training_result.clustered else {}
        ),
        "cluster_model_artifact_paths": (
            dict(cluster_manifest["final_cluster_model_checkpoint_paths"])
            if cluster_manifest is not None
            else {}
        ),
        "clients": [
            {
                "client_id": str(item["client_id"]),
                "candidate_count": int(item["data"].candidate_count),
                "instance_count": int(item["data"].instance_count),
                "raw_pair_count": int(item["data"].pair_count),
                "augmented_pair_count": int(item["data"].augmented_pair_count),
                "split_name": str(item["split_name"]),
                "dataset_indices": [int(value) for value in item["dataset_indices"]],
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
    recommender_type: str = DEFAULT_RECOMMENDER_TYPE,
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
    _require_safe_segment(context_filename, label="context_filename")
    _require_safe_segment(label_filename, label="label_filename")
    artifact_paths = paths or ArtifactPaths()
    run_context = resolve_federated_run_context(paths=artifact_paths, run_id=run_id)
    resolved_model_path = model_path or (
        _recommender_training_dir(
            run_context.run_artifact_dir,
            selection_id,
            persona,
            recommender_type=recommender_type,
        )
        / "model"
        / "global_recommender.npz"
    )
    model = load_recommender(resolved_model_path)
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
        split_name="test",
    )
    if feature_columns is None:
        feature_columns = tuple(loaded_clients[0]["feature_columns"])
    client_metrics: list[dict[str, object]] = []
    for item in loaded_clients:
        candidates = item["candidates"]
        pair_labels = item["pair_labels"]
        score_frame = candidates.loc[:, ["dataset_index", "method_variant"]].copy()
        score_frame["score"] = model.score_candidates(candidates, feature_columns).to_numpy()
        metrics = evaluate_grouped_ranked_scores(
            candidate_scores=score_frame,
            pair_labels=pair_labels,
            top_k=top_k,
        )
        row: dict[str, object] = {
            "client_id": str(item["client_id"]),
            "candidate_count": int(len(candidates)),
            "pair_count": int(len(pair_labels)),
            "instance_count": int(metrics.get("instance_count", 0)),
        }
        aggregate_metrics = metrics.get("aggregate", {})
        if isinstance(aggregate_metrics, Mapping):
            for key, value in aggregate_metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = float(value)
        row["instances"] = metrics.get("instances", [])
        client_metrics.append(row)

    aggregate = _aggregate_client_metrics(client_metrics)
    payload = {
        "status": "evaluated",
        "run_id": run_id,
        "selection_id": selection_id,
        "persona": persona,
        "recommender_type": recommender_type,
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
    split_name: str | None = None,
) -> list[dict[str, Any]]:
    _require_safe_segment(context_filename, label="context_filename")
    _require_safe_segment(label_filename, label="label_filename")
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
        metadata_path = client_dir / "recommender_labels" / selection_id / persona / "simulation_metadata.json"
        if not context_path.exists() or not labels_path.exists():
            if requested_clients is not None:
                missing = context_path if not context_path.exists() else labels_path
                raise FileNotFoundError(f"Missing recommender input for {client_id}: {missing}")
            continue
        candidates = pd.read_parquet(context_path)
        pair_labels = pd.read_parquet(labels_path)
        split_metadata = _load_recommender_split_metadata(metadata_path)
        selected_dataset_indices = _select_recommender_dataset_indices(
            split_metadata=split_metadata,
            split_name=split_name,
        )
        if selected_dataset_indices is not None:
            candidates = _filter_frame_by_dataset_indices(candidates, selected_dataset_indices)
            pair_labels = _filter_frame_by_dataset_indices(pair_labels, selected_dataset_indices)
        if "split" in pair_labels.columns and split_name in {"train", "test"}:
            pair_labels = pair_labels.loc[pair_labels["split"].astype(str) == split_name].copy()
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
                "split_name": split_name or "all",
                "dataset_indices": tuple(sorted(int(value) for value in candidates["dataset_index"].unique())),
                "split_metadata": split_metadata,
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
        cluster_manifest_path=run_dir / "clustering" / "manifest.json",
        cluster_rounds_dir=run_dir / "clustering" / "rounds",
        cluster_models_dir=run_dir / "model" / "clusters",
        completion_marker_path=run_dir / "COMPLETED.json",
    )


def _persist_clustered_training_artifacts(
    *,
    artifacts: RecommenderTrainingArtifacts,
    run_dir: Path,
    training_result: RecommenderSimulationArtifacts,
    config: RecommenderFederatedTrainingConfig,
    feature_columns: Sequence[str],
    model_config: PairwiseLogisticConfig,
) -> dict[str, Any]:
    if not training_result.clustered:
        return {}

    final_round_id = training_result.clustered_rounds[-1].round_id
    round_artifact_paths: list[str] = []
    final_cluster_model_checkpoint_paths: dict[str, str] = {}

    for round_result in training_result.clustered_rounds:
        round_model_dir = artifacts.cluster_models_dir / f"round_{round_result.round_id:04d}"
        cluster_model_checkpoint_paths: dict[str, str] = {}
        for cluster_id, parameters in sorted(round_result.cluster_parameters.items()):
            cluster_model_path = round_model_dir / f"cluster_{cluster_id:03d}.npz"
            cluster_model = create_recommender(
                recommender_type=config.recommender_type,
                n_features=len(feature_columns),
                config=model_config,
            )
            cluster_model.set_parameters(parameters)
            cluster_model.save(cluster_model_path)
            relative_cluster_model_path = str(cluster_model_path.relative_to(run_dir))
            cluster_model_checkpoint_paths[str(cluster_id)] = relative_cluster_model_path
            if round_result.round_id == final_round_id:
                final_cluster_model_checkpoint_paths[str(cluster_id)] = relative_cluster_model_path

        round_payload = {
            "artifact_type": "recommender_cluster_round",
            "run_id": config.run_id,
            "selection_id": config.selection_id,
            "persona": config.persona,
            "recommender_type": config.recommender_type,
            "round_id": int(round_result.round_id),
            "assignments": dict(round_result.assignments),
            "cluster_sizes": dict(round_result.cluster_sizes),
            "projection": dict(round_result.projection_metadata),
            "pca": dict(round_result.projection_metadata),
            "secure_clustering": dict(round_result.secure_clustering_metadata),
            "secure_aggregation_per_cluster": dict(round_result.secure_aggregation_metadata),
            "cluster_model_checkpoint_paths": cluster_model_checkpoint_paths,
        }
        round_artifact_path = artifacts.cluster_rounds_dir / f"round_{round_result.round_id:04d}.json"
        _write_json_atomic(round_artifact_path, round_payload)
        round_artifact_paths.append(str(round_artifact_path.relative_to(run_dir)))

    manifest = {
        "artifact_type": "recommender_cluster_manifest",
        "run_id": config.run_id,
        "selection_id": config.selection_id,
        "persona": config.persona,
        "recommender_type": config.recommender_type,
        "enabled": True,
        "method": config.clustering.method,
        "k": int(config.clustering.k),
        "pca_components": int(config.clustering.pca_components),
        "round_count": int(len(training_result.clustered_rounds)),
        "final_cluster_assignments": dict(training_result.final_cluster_assignments),
        "final_cluster_model_checkpoint_paths": final_cluster_model_checkpoint_paths,
        "round_artifact_paths": round_artifact_paths,
    }
    _write_json_atomic(artifacts.cluster_manifest_path, manifest)
    return manifest


def _evaluate_clustered_recommender_models(
    *,
    loaded_clients: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    cluster_assignments: Mapping[str, int],
    cluster_model_paths: Mapping[str, Path],
    run_id: str,
    selection_id: str,
    persona: str,
    recommender_type: str,
    top_k: Iterable[int] = (1, 3, 5),
    output_path: Path | None = None,
) -> dict[str, Any]:
    model_cache: dict[str, Any] = {}
    client_metrics: list[dict[str, object]] = []
    cluster_metrics: dict[str, list[dict[str, object]]] = {}

    for item in loaded_clients:
        client_id = str(item["client_id"])
        if client_id not in cluster_assignments:
            raise ValueError(f"Missing final cluster assignment for recommender client {client_id!r}.")
        cluster_id = str(cluster_assignments[client_id])
        model_path = cluster_model_paths.get(cluster_id)
        if model_path is None:
            raise ValueError(f"Missing cluster model path for cluster {cluster_id}.")
        model = model_cache.get(cluster_id)
        if model is None:
            model = load_recommender(model_path)
            model_cache[cluster_id] = model
        candidates = item["candidates"]
        pair_labels = item["pair_labels"]
        score_frame = candidates.loc[:, ["dataset_index", "method_variant"]].copy()
        score_frame["score"] = model.score_candidates(candidates, feature_columns).to_numpy()
        metrics = evaluate_grouped_ranked_scores(
            candidate_scores=score_frame,
            pair_labels=pair_labels,
            top_k=top_k,
        )
        row: dict[str, object] = {
            "client_id": client_id,
            "cluster_id": int(cluster_id),
            "cluster_model_path": str(model_path),
            "candidate_count": int(len(candidates)),
            "pair_count": int(len(pair_labels)),
            "instance_count": int(metrics.get("instance_count", 0)),
        }
        aggregate_metrics = metrics.get("aggregate", {})
        if isinstance(aggregate_metrics, Mapping):
            for key, value in aggregate_metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = float(value)
        row["instances"] = metrics.get("instances", [])
        client_metrics.append(row)
        cluster_metrics.setdefault(cluster_id, []).append(row)

    payload = {
        "status": "evaluated_clustered",
        "run_id": run_id,
        "selection_id": selection_id,
        "persona": persona,
        "recommender_type": recommender_type,
        "clustered": True,
        "feature_count": int(len(feature_columns)),
        "client_count": int(len(client_metrics)),
        "generated_at": current_utc_timestamp(),
        "aggregate": _aggregate_client_metrics(client_metrics),
        "clients": client_metrics,
        "clusters": [
            {
                "cluster_id": int(cluster_id),
                "model_path": str(cluster_model_paths[cluster_id]),
                "client_count": int(len(rows)),
                "aggregate": _aggregate_client_metrics(rows),
            }
            for cluster_id, rows in sorted(cluster_metrics.items(), key=lambda item: int(item[0]))
        ],
        "cluster_assignments": {client_id: int(cluster_id) for client_id, cluster_id in cluster_assignments.items()},
        "cluster_model_paths": {cluster_id: str(path) for cluster_id, path in cluster_model_paths.items()},
    }
    if output_path is not None:
        _write_json_atomic(output_path, payload)
    return payload


def _recommender_training_dir(
    run_artifact_dir: Path,
    selection_id: str,
    persona: str,
    *,
    recommender_type: str = DEFAULT_RECOMMENDER_TYPE,
) -> Path:
    return (
        run_artifact_dir
        / "recommender_training"
        / selection_id
        / persona
        / f"{recommender_type}_fedavg"
    )


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
            if not is_recommender_metric_key(key) or not isinstance(value, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value) * weight
            weights[key] = weights.get(key, 0.0) + weight
    return {key: sums[key] / weights[key] for key in sorted(sums) if weights.get(key, 0.0) > 0.0}


def _load_recommender_split_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    instance_split = payload.get("instance_split")
    return instance_split if isinstance(instance_split, dict) else None


def _select_recommender_dataset_indices(
    *,
    split_metadata: Mapping[str, Any] | None,
    split_name: str | None,
) -> tuple[int, ...] | None:
    if split_name not in {"train", "test"} or split_metadata is None:
        return None
    key = "train_dataset_indices" if split_name == "train" else "test_dataset_indices"
    values = split_metadata.get(key)
    if not isinstance(values, list) or not values:
        return None
    return tuple(sorted(int(value) for value in values))


def _filter_frame_by_dataset_indices(
    frame: pd.DataFrame,
    dataset_indices: Sequence[int],
) -> pd.DataFrame:
    if "dataset_index" not in frame.columns:
        return frame
    allowed = {int(value) for value in dataset_indices}
    return frame.loc[frame["dataset_index"].isin(allowed)].copy()


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
