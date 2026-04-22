"""Standalone Stage B federated training orchestration."""

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
    stage_b_completion_marker_path,
    stage_b_global_model_path,
    stage_b_model_metadata_path,
    stage_b_run_manifest_path,
    stage_b_runtime_report_path,
    stage_b_training_history_path,
    stage_b_training_metadata_path,
)
from fed_perso_xai.utils.provenance import (
    build_reproducibility_metadata,
    build_run_id,
    current_utc_timestamp,
    relative_artifact_path,
    resolve_git_commit_hash,
)


@dataclass(frozen=True)
class StageBArtifacts:
    """Stable filesystem pointers for one Stage B run."""

    run_dir: Path
    model_artifact_path: Path
    model_metadata_path: Path
    training_metadata_path: Path
    training_history_path: Path
    runtime_report_path: Path
    completion_marker_path: Path
    config_snapshot_path: Path


def train_federated_stage_b(
    config: FederatedTrainingConfig,
    *,
    run_id: str | None = None,
    partition_data_root: Path | None = None,
    force: bool = False,
) -> tuple[StageBArtifacts, dict[str, Any]]:
    """Run standalone Stage B federated training from persisted client partitions."""

    resolved_run_id = run_id or build_run_id(
        experiment_type="federated-stage-b",
        dataset_name=config.dataset_name,
        seed=config.seed,
        num_clients=config.num_clients,
        alpha=config.alpha,
    )
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
    artifacts = _build_stage_b_artifacts(result_dir)
    config_snapshot = config.to_dict()
    config_hash = _stable_json_sha256(config_snapshot)

    if (
        _is_completed_stage_b_run(
            artifacts,
            expected_run_id=resolved_run_id,
            expected_config_hash=config_hash,
        )
        and not force
    ):
        metadata = json.loads(artifacts.training_metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "skipped_existing"
        metadata["skipped"] = True
        return artifacts, metadata
    if artifacts.completion_marker_path.exists():
        artifacts.completion_marker_path.unlink()

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
        raise ValueError("No client datasets were loaded for Stage B training.")

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

    started_at = current_utc_timestamp()
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
        "artifact_version": "stage_b_model_v1",
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
            "assumption": "Frozen shared preprocessing fitted once during Stage A.",
        },
        "partition_reference": {
            "partition_root": str(resolved_partition_root),
            "partition_metadata_path": str(partition_metadata_path),
            "num_clients": int(config.num_clients),
            "alpha": float(config.alpha),
        },
    }
    artifacts.model_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.model_metadata_path.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")

    training_metadata = {
        "stage": "stage_b",
        "status": "completed",
        "run_id": resolved_run_id,
        "dataset_name": config.dataset_name,
        "partition_data_root": str(resolved_partition_root),
        "partition_metadata_path": str(partition_metadata_path),
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
        "stage_success": True,
        "round_history_summary": training_result.round_history,
    }
    artifacts.training_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.training_metadata_path.write_text(
        json.dumps(training_metadata, indent=2),
        encoding="utf-8",
    )

    manifest = _build_stage_b_manifest(
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
    stage_b_manifest_path = stage_b_run_manifest_path(result_dir)
    stage_b_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    artifacts.completion_marker_path.write_text(
        json.dumps(
            {
                "stage": "stage_b",
                "status": "completed",
                "run_id": resolved_run_id,
                "completed_at": completed_at,
                "training_metadata_path": relative_artifact_path(
                    artifacts.training_metadata_path,
                    result_dir,
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifacts, training_metadata


def _build_stage_b_artifacts(run_dir: Path) -> StageBArtifacts:
    return StageBArtifacts(
        run_dir=run_dir,
        model_artifact_path=stage_b_global_model_path(run_dir),
        model_metadata_path=stage_b_model_metadata_path(run_dir),
        training_metadata_path=stage_b_training_metadata_path(run_dir),
        training_history_path=stage_b_training_history_path(run_dir),
        runtime_report_path=stage_b_runtime_report_path(run_dir),
        completion_marker_path=stage_b_completion_marker_path(run_dir),
        config_snapshot_path=run_dir / "config_snapshot.json",
    )


def _is_completed_stage_b_run(
    artifacts: StageBArtifacts,
    *,
    expected_run_id: str,
    expected_config_hash: str,
) -> bool:
    if not (
        artifacts.model_artifact_path.exists()
        and artifacts.model_metadata_path.exists()
        and artifacts.training_metadata_path.exists()
        and artifacts.completion_marker_path.exists()
    ):
        return False
    metadata = json.loads(artifacts.training_metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("status") == "completed"
        and bool(metadata.get("stage_success"))
        and metadata.get("run_id") == expected_run_id
        and metadata.get("training_config_sha256") == expected_config_hash
    )


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


def _build_stage_b_manifest(
    *,
    run_dir: Path,
    run_id: str,
    dataset_name: str,
    config: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "manifest_version": "stage_b_run_manifest_v1",
        "run_id": run_id,
        "mode": "federated_stage_b",
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
