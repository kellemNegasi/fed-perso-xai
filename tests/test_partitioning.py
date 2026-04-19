from __future__ import annotations

import numpy as np

from fed_perso_xai.data.partitioning import dirichlet_partition_labels


def test_dirichlet_partition_is_reproducible_and_complete() -> None:
    y = np.asarray([0] * 60 + [1] * 40, dtype=np.int64)
    partitions_a = dirichlet_partition_labels(
        y=y,
        num_clients=5,
        alpha=0.8,
        seed=123,
        min_client_samples=5,
        max_retries=20,
    )
    partitions_b = dirichlet_partition_labels(
        y=y,
        num_clients=5,
        alpha=0.8,
        seed=123,
        min_client_samples=5,
        max_retries=20,
    )

    assert [partition.tolist() for partition in partitions_a] == [
        partition.tolist() for partition in partitions_b
    ]
    concatenated = np.concatenate(partitions_a)
    assert sorted(concatenated.tolist()) == list(range(y.shape[0]))
    assert len(np.unique(concatenated)) == y.shape[0]
    assert all(partition.shape[0] >= 5 for partition in partitions_a)
