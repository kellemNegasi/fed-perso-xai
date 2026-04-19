"""Filesystem path helpers for datasets and training artifacts."""

from __future__ import annotations

from pathlib import Path


def format_alpha(alpha: float) -> str:
    """Format a Dirichlet alpha for stable directory naming."""

    return str(float(alpha))


def partition_root(base_dir: Path, num_clients: int, alpha: float) -> Path:
    """Return the directory containing all client partitions."""

    return base_dir / f"{num_clients}_clients" / f"alpha_{format_alpha(alpha)}"


def client_dir(base_dir: Path, num_clients: int, alpha: float, client_id: int) -> Path:
    """Return the directory for one client partition."""

    return partition_root(base_dir, num_clients, alpha) / f"client_{client_id}"


def training_run_dir(
    results_root: Path,
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
) -> Path:
    """Return the result directory for one training run."""

    return (
        results_root
        / dataset_name
        / f"{num_clients}_clients"
        / f"alpha_{format_alpha(alpha)}"
        / f"seed_{seed}"
    )
