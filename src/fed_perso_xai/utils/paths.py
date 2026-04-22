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


def stage_b_model_dir(run_dir: Path) -> Path:
    """Return the directory containing frozen Stage B model artifacts."""

    return run_dir / "model"


def stage_b_training_dir(run_dir: Path) -> Path:
    """Return the directory containing Stage B training metadata artifacts."""

    return run_dir / "training"


def stage_b_global_model_path(run_dir: Path) -> Path:
    """Return the frozen global model artifact path."""

    return stage_b_model_dir(run_dir) / "global_model.npz"


def stage_b_model_metadata_path(run_dir: Path) -> Path:
    """Return the Stage B model metadata path."""

    return stage_b_model_dir(run_dir) / "model_metadata.json"


def stage_b_training_metadata_path(run_dir: Path) -> Path:
    """Return the Stage B training metadata path."""

    return stage_b_training_dir(run_dir) / "training_metadata.json"


def stage_b_training_history_path(run_dir: Path) -> Path:
    """Return the Stage B training history CSV path."""

    return stage_b_training_dir(run_dir) / "training_history.csv"


def stage_b_runtime_report_path(run_dir: Path) -> Path:
    """Return the Stage B runtime report path."""

    return stage_b_training_dir(run_dir) / "runtime_report.json"


def stage_b_completion_marker_path(run_dir: Path) -> Path:
    """Return the Stage B completion marker path."""

    return stage_b_training_dir(run_dir) / "stage_b.done"


def stage_b_run_manifest_path(run_dir: Path) -> Path:
    """Return the Stage B manifest path."""

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
