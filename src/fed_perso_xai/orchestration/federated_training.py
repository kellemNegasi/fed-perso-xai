"""Federated training orchestration from persisted client partitions."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fed_perso_xai.data.serialization import copy_shared_artifacts, load_client_datasets
from fed_perso_xai.fl.client import ClientData
from fed_perso_xai.fl.simulation import run_federated_training
from fed_perso_xai.models import create_model
from fed_perso_xai.models.persistence import save_global_model_parameters
from fed_perso_xai.utils.config import FederatedTrainingConfig
from fed_perso_xai.utils.paths import (
    federated_run_dir,
    partition_root,
    federated_completion_marker_path,
    federated_model_path,
    federated_model_metadata_path,
    federated_run_artifact_dir,
    federated_run_metadata_path,
    federated_run_manifest_path,
    federated_runtime_report_path,
    federated_training_history_path,
    federated_training_metadata_path,
)
from fed_perso_xai.utils.provenance import (
    build_reproducibility_metadata,
    build_run_id,
    current_utc_timestamp,
    relative_artifact_path,
    resolve_git_commit_hash,
)


@dataclass(frozen=True)
class FederatedTrainingArtifacts:
    """Stable filesystem pointers for one federated training run."""

    run_dir: Path
    model_artifact_path: Path
    model_metadata_path: Path
    training_metadata_path: Path
    training_history_path: Path
    runtime_report_path: Path
    completion_marker_path: Path
    config_snapshot_path: Path


def train_federated_from_partitions(
    config: FederatedTrainingConfig,
    *,
    run_id: str | None = None,
    partition_data_root: Path | None = None,
    force: bool = False,
) -> tuple[FederatedTrainingArtifacts, dict[str, Any]]:
    """Train a federated model from persisted client partitions."""

    resolved_partition_root = partition_data_root or partition_root(
        config.paths.partition_root,
        config.dataset_name,
        config.num_clients,
        config.alpha,
        config.seed,
    )
    result_dir = federated_run_dir(
        config.paths,
        config.dataset_name,
        config.num_clients,
        config.alpha,
        config.seed,
    )
    artifacts = _build_federated_training_artifacts(result_dir)
    config_snapshot = config.to_dict()
    config_hash = _stable_json_sha256(config_snapshot)
    run_started_at = current_utc_timestamp()

    partition_metadata_path = _resolve_partition_metadata_path(resolved_partition_root)
    if not partition_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing prepared partition metadata at '{partition_metadata_path}'. Run prepare-data first."
        )

    partition_metadata = json.loads(partition_metadata_path.read_text(encoding="utf-8"))
    _validate_partition_metadata(
        metadata=partition_metadata,
        config=config,
        metadata_path=partition_metadata_path,
    )
    partition_signature = _build_partition_source_signature(
        root_dir=resolved_partition_root,
        partition_metadata=partition_metadata,
    )

    existing_metadata = _load_completed_federated_training_metadata(artifacts)
    if run_id is None and existing_metadata is not None and not force:
        mismatches = _describe_federated_training_reuse_mismatches(
            metadata=existing_metadata,
            expected_run_id=str(existing_metadata.get("run_id", "")),
            expected_config_hash=config_hash,
            expected_partition_signature=partition_signature,
        )
        if not mismatches:
            metadata = dict(existing_metadata)
            metadata["status"] = "skipped_existing"
            metadata["skipped"] = True
            return artifacts, metadata

    resolved_run_id = run_id or build_run_id(
        experiment_type="federated-training",
        dataset_name=config.dataset_name,
        seed=config.seed,
        num_clients=config.num_clients,
        alpha=config.alpha,
        model_name=config.model_name,
        timestamp=run_started_at,
        run_defining_payload={
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "model": config_snapshot.get("model"),
            "num_clients": config.num_clients,
            "alpha": config.alpha,
            "rounds": config.rounds,
            "strategy_name": config.strategy_name,
            "simulation_backend": config.simulation_backend,
            "seed": config.seed,
            "secure_aggregation": config.secure_aggregation,
        },
    )

    if existing_metadata is not None:
        mismatches = _describe_federated_training_reuse_mismatches(
            metadata=existing_metadata,
            expected_run_id=resolved_run_id,
            expected_config_hash=config_hash,
            expected_partition_signature=partition_signature,
        )
        if not mismatches and not force:
            metadata = dict(existing_metadata)
            metadata["status"] = "skipped_existing"
            metadata["skipped"] = True
            return artifacts, metadata
        if mismatches and not force:
            mismatch_text = "; ".join(mismatches)
            raise FileExistsError(
                "A completed federated training run already exists at "
                f"'{result_dir}' with different inputs ({mismatch_text}). "
                "Re-run with force=True / --force to overwrite that completed run."
            )
    if artifacts.completion_marker_path.exists():
        artifacts.completion_marker_path.unlink()

    prepared_root = Path(str(partition_metadata["prepared_root"]))
    client_datasets = [
        ClientData(
            client_id=client.client_id,
            X_train=client.train.X,
            y_train=client.train.y,
            row_ids_train=client.train.row_ids,
            X_test=client.test.X,
            y_test=client.test.y,
            row_ids_test=client.test.row_ids,
        )
        for client in load_client_datasets(resolved_partition_root, config.num_clients)
    ]
    if not client_datasets:
        raise ValueError("No client datasets were loaded for federated training.")
    # TODO: Remove or invalidate legacy comparison inputs when starting a new run
    # to prevent stale-result reports during reruns and forced reruns.
    result_dir.mkdir(parents=True, exist_ok=True)
    copy_shared_artifacts(prepared_root, result_dir)
    shutil.copy2(partition_metadata_path, result_dir / "partition_metadata.json")
    config_snapshot_path = artifacts.config_snapshot_path
    config_snapshot_path.write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")
    reproducibility_path = result_dir / "reproducibility_metadata.json"
    reproducibility_path.write_text(
        json.dumps(build_reproducibility_metadata(seed=config.seed), indent=2),
        encoding="utf-8",
    )

    started_at = run_started_at
    training_result = run_federated_training(
        client_datasets=client_datasets,
        config=config,
    )
    completed_at = current_utc_timestamp()

    model = create_model(
        config.model_name,
        n_features=client_datasets[0].X_train.shape[1],
        config=config.model,
    )
    model.set_parameters(training_result.final_parameters)
    model_artifact_path = save_global_model_parameters(artifacts.model_artifact_path, model)

    artifacts.runtime_report_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.runtime_report_path.write_text(
        json.dumps(training_result.runtime_report, indent=2),
        encoding="utf-8",
    )
    _write_training_history_csv(artifacts.training_history_path, training_result.round_history)

    model_metadata = {
        "artifact_type": "frozen_global_model",
        "artifact_version": "federated_model_v1",
        "run_id": resolved_run_id,
        "model_type": config.model_name,
        "model_config": dict(config_snapshot.get("model") or {}),
        "n_features": int(client_datasets[0].X_train.shape[1]),
        "parameter_count": int(len(training_result.final_parameters)),
        "class_labels": [0, 1],
        "serialization_format": "numpy_parameter_bundle_npz_v1",
        "model_artifact_path": relative_artifact_path(model_artifact_path, result_dir),
        "training_metadata_path": relative_artifact_path(artifacts.training_metadata_path, result_dir),
        "config_snapshot_path": relative_artifact_path(config_snapshot_path, result_dir),
        "preprocessing": {
            "preprocessor_artifact": _relative_if_exists(result_dir / "preprocessor.joblib", result_dir),
            "feature_metadata_artifact": _relative_if_exists(result_dir / "feature_metadata.json", result_dir),
            "prepared_root": str(prepared_root),
            "assumption": "Frozen shared preprocessing fitted once during data preparation.",
        },
        "partition_reference": {
            "partition_root": partition_signature["partition_data_root"],
            "partition_metadata_path": str(partition_metadata_path),
            "partition_metadata_sha256": partition_signature["partition_metadata_sha256"],
            "num_clients": int(config.num_clients),
            "alpha": float(config.alpha),
            "feature_metadata_path": str(partition_metadata.get("feature_metadata_path", "")),
        },
    }
    artifacts.model_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.model_metadata_path.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")

    training_metadata = {
        "mode": "federated_training",
        "status": "completed",
        "run_id": resolved_run_id,
        "dataset_name": config.dataset_name,
        "partition_data_root": partition_signature["partition_data_root"],
        "partition_metadata_path": str(partition_metadata_path),
        "partition_metadata_sha256": partition_signature["partition_metadata_sha256"],
        "prepared_root": str(prepared_root),
        "num_clients": int(config.num_clients),
        "alpha": float(config.alpha),
        "model_type": config.model_name,
        "training_config": config_snapshot,
        "training_config_sha256": config_hash,
        "seed_values": {
            "global_seed": int(config.seed),
            "secure_seed": int(config.secure_seed),
        },
        "started_at": started_at,
        "completed_at": completed_at,
        "rounds_requested": int(config.rounds),
        "rounds_completed": int(len(training_result.round_history)),
        "simulation_backend_requested": config.simulation_backend,
        "simulation_backend_actual": training_result.actual_backend,
        "runtime_report_path": relative_artifact_path(artifacts.runtime_report_path, result_dir),
        "model_artifact_path": relative_artifact_path(model_artifact_path, result_dir),
        "model_metadata_path": relative_artifact_path(artifacts.model_metadata_path, result_dir),
        "training_history_path": relative_artifact_path(artifacts.training_history_path, result_dir),
        "completion_marker_path": relative_artifact_path(artifacts.completion_marker_path, result_dir),
        "force_requested": bool(force),
        "client_count_loaded": int(len(client_datasets)),
        "partition_client_count": int(len(partition_metadata.get("clients", []))),
        "training_success": True,
        "round_history_summary": training_result.round_history,
    }
    artifacts.training_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.training_metadata_path.write_text(
        json.dumps(training_metadata, indent=2),
        encoding="utf-8",
    )

    manifest = _build_federated_training_manifest(
        run_dir=result_dir,
        run_id=resolved_run_id,
        dataset_name=config.dataset_name,
        config=config_snapshot,
        artifact_paths={
            "config_snapshot": config_snapshot_path,
            "model": model_artifact_path,
            "model_metadata": artifacts.model_metadata_path,
            "training_metadata": artifacts.training_metadata_path,
            "training_history": artifacts.training_history_path,
            "runtime_report": artifacts.runtime_report_path,
            "preprocessor": result_dir / "preprocessor.joblib",
            "feature_metadata": result_dir / "feature_metadata.json",
            "dataset_metadata": result_dir / "dataset_metadata.json",
            "split_metadata": result_dir / "split_metadata.json",
            "partition_metadata": result_dir / "partition_metadata.json",
            "reproducibility_metadata": reproducibility_path,
        },
    )
    manifest_path = federated_run_manifest_path(result_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_artifact_dir = _sync_federated_run_artifact(
        paths=config.paths,
        run_id=resolved_run_id,
        canonical_run_dir=result_dir,
        partition_metadata=partition_metadata,
        training_metadata=training_metadata,
        model_metadata=model_metadata,
        artifact_paths={
            "config_snapshot": config_snapshot_path,
            "model_artifact": model_artifact_path,
            "model_metadata": artifacts.model_metadata_path,
            "training_metadata": artifacts.training_metadata_path,
            "training_history": artifacts.training_history_path,
            "runtime_report": artifacts.runtime_report_path,
            "run_manifest": manifest_path,
            "preprocessor": result_dir / "preprocessor.joblib",
            "feature_metadata": result_dir / "feature_metadata.json",
            "dataset_metadata": result_dir / "dataset_metadata.json",
            "split_metadata": result_dir / "split_metadata.json",
            "partition_metadata": result_dir / "partition_metadata.json",
            "reproducibility_metadata": reproducibility_path,
        },
    )

    artifacts.completion_marker_path.write_text(
        json.dumps(
            {
                "mode": "federated_training",
                "status": "completed",
                "run_id": resolved_run_id,
                "completed_at": completed_at,
                "training_metadata_path": relative_artifact_path(
                    artifacts.training_metadata_path,
                    result_dir,
                ),
                "run_artifact_dir": str(run_artifact_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifacts, training_metadata


def _build_federated_training_artifacts(run_dir: Path) -> FederatedTrainingArtifacts:
    return FederatedTrainingArtifacts(
        run_dir=run_dir,
        model_artifact_path=federated_model_path(run_dir),
        model_metadata_path=federated_model_metadata_path(run_dir),
        training_metadata_path=federated_training_metadata_path(run_dir),
        training_history_path=federated_training_history_path(run_dir),
        runtime_report_path=federated_runtime_report_path(run_dir),
        completion_marker_path=federated_completion_marker_path(run_dir),
        config_snapshot_path=run_dir / "config_snapshot.json",
    )


def _load_completed_federated_training_metadata(
    artifacts: FederatedTrainingArtifacts,
) -> dict[str, Any] | None:
    if not (
        artifacts.model_artifact_path.exists()
        and artifacts.model_metadata_path.exists()
        and artifacts.training_metadata_path.exists()
        and artifacts.completion_marker_path.exists()
    ):
        return None
    # TODO this load could fail if the json is truncated or invalid, which could happen if a previous run was interrupted during writing. 
    # We should detect and handle that case more gracefully, perhaps by treating it as an incomplete run and allowing a new run to proceed without force=True.
    metadata = json.loads(artifacts.training_metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or not bool(metadata.get("training_success")):
        return None
    return metadata


def _describe_federated_training_reuse_mismatches(
    *,
    metadata: dict[str, Any],
    expected_run_id: str,
    expected_config_hash: str,
    expected_partition_signature: dict[str, str],
) -> list[str]:
    mismatches: list[str] = []
    if metadata.get("run_id") != expected_run_id:
        mismatches.append("run_id")
    if metadata.get("training_config_sha256") != expected_config_hash:
        mismatches.append("training_config_sha256")
    if metadata.get("partition_data_root") != expected_partition_signature["partition_data_root"]:
        mismatches.append("partition_data_root")
    if metadata.get("partition_metadata_sha256") != expected_partition_signature["partition_metadata_sha256"]:
        mismatches.append("partition_metadata_sha256")
    return mismatches


def _resolve_partition_metadata_path(root_dir: Path) -> Path:
    explicit = root_dir / "partition_metadata.json"
    if explicit.exists():
        return explicit
    return root_dir / "metadata.json"


def _validate_partition_metadata(
    *,
    metadata: dict[str, Any],
    config: FederatedTrainingConfig,
    metadata_path: Path,
) -> None:
    expected_values = {
        "dataset_name": config.dataset_name,
        "seed": config.seed,
        "num_clients": config.num_clients,
        "alpha": config.alpha,
    }
    mismatches = []
    for field_name, expected in expected_values.items():
        actual = metadata.get(field_name)
        if actual != expected:
            mismatches.append(f"{field_name}: expected {expected!r}, found {actual!r}")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            f"Prepared partition metadata at '{metadata_path}' does not match the requested run: {details}."
        )


def _build_partition_source_signature(
    *,
    root_dir: Path,
    partition_metadata: dict[str, Any],
) -> dict[str, str]:
    return {
        "partition_data_root": str(root_dir.resolve()),
        "partition_metadata_sha256": _stable_json_sha256(partition_metadata),
    }


def _write_training_history_csv(path: Path, round_history: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "round",
        "evaluate_loss",
        "aggregation_mode",
        "aggregation_num_contributors",
        "aggregation_helper_count",
        "fit_metrics_json",
        "evaluate_metrics_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in round_history:
            aggregation = record.get("aggregation") or {}
            writer.writerow(
                {
                    "round": record.get("round"),
                    "evaluate_loss": record.get("evaluate_loss"),
                    "aggregation_mode": aggregation.get("mode"),
                    "aggregation_num_contributors": aggregation.get("num_contributors"),
                    "aggregation_helper_count": aggregation.get("helper_count"),
                    "fit_metrics_json": json.dumps(record.get("fit_metrics") or {}, sort_keys=True),
                    "evaluate_metrics_json": json.dumps(
                        record.get("evaluate_metrics") or {},
                        sort_keys=True,
                    ),
                }
            )
    return path


def _build_federated_training_manifest(
    *,
    run_dir: Path,
    run_id: str,
    dataset_name: str,
    config: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "manifest_version": "federated_run_manifest_v1",
        "run_id": run_id,
        "mode": "federated_training",
        "dataset_name": dataset_name,
        "timestamp": current_utc_timestamp(),
        "git_commit_hash": resolve_git_commit_hash(run_dir),
        "important_config": {
            "seed": config["seed"],
            "dataset_name": config["dataset_name"],
            "model_name": config["model_name"],
            "model": config["model"],
            "num_clients": config["num_clients"],
            "alpha": config["alpha"],
            "strategy_name": config["strategy_name"],
            "rounds": config["rounds"],
            "simulation_backend": config["simulation_backend"],
        },
        "artifacts": {
            key: _relative_if_exists(path, run_dir)
            for key, path in artifact_paths.items()
        },
    }


def _relative_if_exists(path: Path, run_dir: Path) -> str | None:
    if not path.exists():
        return None
    return relative_artifact_path(path, run_dir)


def _stable_json_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sync_federated_run_artifact(
    *,
    paths: Any,
    run_id: str,
    canonical_run_dir: Path,
    partition_metadata: dict[str, Any],
    training_metadata: dict[str, Any],
    model_metadata: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> Path:
    run_artifact_dir = federated_run_artifact_dir(paths, run_id)
    if run_artifact_dir.exists():
        shutil.rmtree(run_artifact_dir)
    run_artifact_dir.mkdir(parents=True, exist_ok=True)

    for key, source in artifact_paths.items():
        if not source.exists():
            continue
        if key == "model_artifact":
            destination = run_artifact_dir / "model" / "global_model.npz"
        elif key == "model_metadata":
            destination = run_artifact_dir / "model" / "model_metadata.json"
        elif key == "training_metadata":
            destination = run_artifact_dir / "training" / "training_metadata.json"
        elif key == "training_history":
            destination = run_artifact_dir / "training" / "training_history.csv"
        elif key == "runtime_report":
            destination = run_artifact_dir / "training" / "runtime_report.json"
        elif key == "run_manifest":
            destination = run_artifact_dir / "run_manifest.json"
        else:
            destination = run_artifact_dir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    run_metadata = {
        "artifact_type": "federated_run",
        "artifact_version": "federated_run_v2",
        "run_id": run_id,
        "created_at": training_metadata["started_at"],
        "completed_at": training_metadata["completed_at"],
        "status": training_metadata["status"],
        "model_type": training_metadata["model_type"],
        "dataset_name": training_metadata["dataset_name"],
        "training_config": training_metadata["training_config"],
        "training_config_sha256": training_metadata["training_config_sha256"],
        "seed_values": training_metadata["seed_values"],
        "canonical_run_dir": str(canonical_run_dir),
        "model_artifact_path": "model/global_model.npz",
        "model_metadata_path": "model/model_metadata.json",
        "training_metadata_path": "training/training_metadata.json",
        "run_manifest_path": "run_manifest.json",
        "feature_metadata_path": "feature_metadata.json" if (run_artifact_dir / "feature_metadata.json").exists() else None,
        "partition_reference": {
            "partition_data_root": training_metadata["partition_data_root"],
            "partition_metadata_path": training_metadata["partition_metadata_path"],
            "partition_metadata_sha256": training_metadata["partition_metadata_sha256"],
            "feature_metadata_path": partition_metadata.get("feature_metadata_path"),
            "prepared_root": training_metadata["prepared_root"],
            "num_clients": training_metadata["num_clients"],
            "alpha": training_metadata["alpha"],
        },
        "model_summary": {
            "n_features": model_metadata["n_features"],
            "parameter_count": model_metadata["parameter_count"],
            "serialization_format": model_metadata["serialization_format"],
            "class_labels": model_metadata["class_labels"],
        },
        "artifacts": {
            key: str(path.relative_to(run_artifact_dir))
            for key, path in {
                key: (
                    run_artifact_dir / "model" / "global_model.npz"
                    if key == "model_artifact"
                    else run_artifact_dir / "model" / "model_metadata.json"
                    if key == "model_metadata"
                    else run_artifact_dir / "training" / "training_metadata.json"
                    if key == "training_metadata"
                    else run_artifact_dir / "training" / "training_history.csv"
                    if key == "training_history"
                    else run_artifact_dir / "training" / "runtime_report.json"
                    if key == "runtime_report"
                    else run_artifact_dir / "run_manifest.json"
                    if key == "run_manifest"
                    else run_artifact_dir / source.name
                )
                for key, source in artifact_paths.items()
                if (
                    (run_artifact_dir / "model" / "global_model.npz").exists()
                    if key == "model_artifact"
                    else (run_artifact_dir / "model" / "model_metadata.json").exists()
                    if key == "model_metadata"
                    else (run_artifact_dir / "training" / "training_metadata.json").exists()
                    if key == "training_metadata"
                    else (run_artifact_dir / "training" / "training_history.csv").exists()
                    if key == "training_history"
                    else (run_artifact_dir / "training" / "runtime_report.json").exists()
                    if key == "runtime_report"
                    else (run_artifact_dir / "run_manifest.json").exists()
                    if key == "run_manifest"
                    else (run_artifact_dir / source.name).exists()
                )
            }
        },
    }
    federated_run_metadata_path(run_artifact_dir).write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )
    return run_artifact_dir
