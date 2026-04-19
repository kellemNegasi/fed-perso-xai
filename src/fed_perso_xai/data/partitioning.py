"""Dirichlet partitioning and client-level train/test split helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ClientSplit:
    """Processed local train/test arrays for one client."""

    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    row_ids_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    row_ids_test: np.ndarray

    @property
    def train_size(self) -> int:
        return int(self.y_train.shape[0])

    @property
    def test_size(self) -> int:
        return int(self.y_test.shape[0])


def dirichlet_partition_labels(
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_client_samples: int,
    max_retries: int,
) -> list[np.ndarray]:
    """Partition labels into clients using a class-aware Dirichlet split."""

    if num_clients < 2:
        raise ValueError("num_clients must be at least 2.")
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive.")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if len(y) < num_clients:
        raise ValueError("Number of samples must be at least the number of clients.")

    rng = np.random.default_rng(seed)
    class_labels = np.unique(y)

    for _ in range(max_retries):
        partitions: list[list[int]] = [[] for _ in range(num_clients)]
        for class_label in class_labels:
            class_indices = np.where(y == class_label)[0]
            shuffled = rng.permutation(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            counts = rng.multinomial(shuffled.shape[0], proportions)
            boundaries = np.cumsum(counts[:-1])
            for client_id, chunk in enumerate(np.split(shuffled, boundaries)):
                partitions[client_id].extend(int(index) for index in chunk)

        partition_arrays = [
            np.asarray(sorted(client_indices), dtype=np.int64)
            for client_indices in partitions
        ]
        if all(indices.shape[0] >= min_client_samples for indices in partition_arrays):
            return partition_arrays

    sizes = [len(client_indices) for client_indices in partitions]
    raise RuntimeError(
        "Failed to create a valid Dirichlet partition after retries. "
        f"Last client sizes: {sizes}"
    )


def split_client_partition(
    X: np.ndarray,
    y: np.ndarray,
    row_ids: np.ndarray,
    client_id: int,
    test_size: float,
    seed: int,
) -> ClientSplit:
    """Split one client's partition into local train/test subsets."""

    indices = np.arange(y.shape[0], dtype=np.int64)
    stratify = y if _can_stratify(y, test_size) else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed + client_id,
        stratify=stratify,
    )
    return ClientSplit(
        client_id=client_id,
        X_train=X[train_idx],
        y_train=y[train_idx],
        row_ids_train=row_ids[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        row_ids_test=row_ids[test_idx],
    )


def summarize_labels(y: np.ndarray) -> dict[str, int]:
    """Return a compact label-count summary."""

    counts = Counter(int(label) for label in y.tolist())
    return {str(label): count for label, count in sorted(counts.items())}


def _can_stratify(y: np.ndarray, test_size: float) -> bool:
    if y.shape[0] < 2:
        return False
    unique, counts = np.unique(y, return_counts=True)
    if unique.shape[0] < 2:
        return False
    if np.any(counts < 2):
        return False
    estimated_test_count = max(1, int(round(y.shape[0] * test_size)))
    return estimated_test_count >= unique.shape[0]
