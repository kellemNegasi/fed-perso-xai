"""Filesystem path helpers for prepared data and experiment outputs."""

from __future__ import annotations

from pathlib import Path

from fed_perso_xai.utils.config import ArtifactPaths


def format_alpha(alpha: float) -> str:
    """Format a Dirichlet alpha for stable directory naming."""

    return str(float(alpha))


def prepared_dir(paths: ArtifactPaths, dataset_name: str, seed: int) -> Path:
    """Return the directory containing shared prepared-data artifacts."""

    return paths.prepared_root / dataset_name / f"seed_{seed}"


def partition_root(
    base_dir: Path,
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
) -> Path:
    """Return the directory containing all client partitions."""

    return (
        base_dir
        / dataset_name
        / f"{num_clients}_clients"
        / f"alpha_{format_alpha(alpha)}"
        / f"seed_{seed}"
    )


def client_dir(
    base_dir: Path,
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
    client_id: int,
) -> Path:
    """Return the directory for one client partition."""

    return partition_root(base_dir, dataset_name, num_clients, alpha, seed) / f"client_{client_id}"


def centralized_run_dir(paths: ArtifactPaths, dataset_name: str, seed: int) -> Path:
    """Return the result directory for one centralized baseline run."""

    return paths.centralized_root / dataset_name / f"seed_{seed}"


def federated_run_dir(
    paths: ArtifactPaths,
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
) -> Path:
    """Return the result directory for one federated baseline run."""

    return (
        paths.federated_root
        / dataset_name
        / f"{num_clients}_clients"
        / f"alpha_{format_alpha(alpha)}"
        / f"seed_{seed}"
    )


def federated_runs_root(paths: ArtifactPaths) -> Path:
    """Return the stable registry root keyed by run_id."""

    return paths.federated_root / "runs"


def federated_run_artifact_dir(paths: ArtifactPaths, run_id: str) -> Path:
    """Return the run-addressable artifact directory for one federated run."""

    return federated_runs_root(paths) / run_id


def federated_run_metadata_path(run_artifact_dir: Path) -> Path:
    """Return the run-level metadata path inside a run-addressable artifact root."""

    return run_artifact_dir / "run_metadata.json"


def federated_client_artifact_dir(run_artifact_dir: Path, client_id: str) -> Path:
    """Return the client-scoped artifact directory under one run."""

    return run_artifact_dir / "clients" / client_id


def federated_client_metadata_path(run_artifact_dir: Path, client_id: str) -> Path:
    """Return the client-level metadata path under one run."""

    return federated_client_artifact_dir(run_artifact_dir, client_id) / "client_metadata.json"


def federated_shard_artifact_dir(
    run_artifact_dir: Path,
    client_id: str,
    split: str,
    shard_id: str,
) -> Path:
    """Return the shard-scoped artifact directory for one client split shard."""

    return federated_client_artifact_dir(run_artifact_dir, client_id) / f"{split}_shards" / shard_id


def federated_shard_metadata_path(
    run_artifact_dir: Path,
    client_id: str,
    split: str,
    shard_id: str,
) -> Path:
    """Return the shard-level metadata path."""

    return federated_shard_artifact_dir(run_artifact_dir, client_id, split, shard_id) / "shard_metadata.json"


def federated_detailed_explanations_dir(
    run_artifact_dir: Path,
    client_id: str,
    split: str,
    shard_id: str,
    explainer_name: str,
) -> Path:
    """Return the explainer-specific directory for detailed explanation outputs."""

    return (
        federated_shard_artifact_dir(run_artifact_dir, client_id, split, shard_id)
        / "detailed_explanations"
        / explainer_name
    )


def federated_metrics_results_dir(
    run_artifact_dir: Path,
    client_id: str,
    split: str,
    shard_id: str,
    explainer_name: str,
) -> Path:
    """Return the explainer-specific directory for metric outputs."""

    return (
        federated_shard_artifact_dir(run_artifact_dir, client_id, split, shard_id)
        / "metrics_results"
        / explainer_name
    )


def federated_job_status_dir(
    run_artifact_dir: Path,
    client_id: str,
    split: str,
    shard_id: str,
) -> Path:
    """Return the shard-local status directory for explain/evaluate jobs."""

    return federated_shard_artifact_dir(run_artifact_dir, client_id, split, shard_id) / "_status"


def federated_model_dir(run_dir: Path) -> Path:
    """Return the directory containing frozen federated model artifacts."""

    return run_dir / "model"


def federated_training_dir(run_dir: Path) -> Path:
    """Return the directory containing federated training metadata artifacts."""

    return run_dir / "training"


def federated_model_path(run_dir: Path) -> Path:
    """Return the frozen global model artifact path."""

    return federated_model_dir(run_dir) / "global_model.npz"


def federated_model_metadata_path(run_dir: Path) -> Path:
    """Return the federated model metadata path."""

    return federated_model_dir(run_dir) / "model_metadata.json"


def federated_training_metadata_path(run_dir: Path) -> Path:
    """Return the federated training metadata path."""

    return federated_training_dir(run_dir) / "training_metadata.json"


def federated_training_history_path(run_dir: Path) -> Path:
    """Return the federated training history CSV path."""

    return federated_training_dir(run_dir) / "training_history.csv"


def federated_runtime_report_path(run_dir: Path) -> Path:
    """Return the federated runtime report path."""

    return federated_training_dir(run_dir) / "runtime_report.json"


def federated_completion_marker_path(run_dir: Path) -> Path:
    """Return the federated training completion marker path."""

    return federated_training_dir(run_dir) / "training.done"


def federated_run_manifest_path(run_dir: Path) -> Path:
    """Return the federated training manifest path."""

    return run_dir / "run_manifest.json"


def comparison_run_dir(
    paths: ArtifactPaths,
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
) -> Path:
    """Return the output directory for baseline comparison reports."""

    return (
        paths.comparison_root
        / dataset_name
        / f"{num_clients}_clients"
        / f"alpha_{format_alpha(alpha)}"
        / f"seed_{seed}"
    )
