from __future__ import annotations

import numpy as np

from fed_perso_xai.data.partitioning import dirichlet_partition_labels, split_client_partition


def test_partitioning_reproducibility_and_split_generation() -> None:
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

    X = np.arange(100 * 4, dtype=np.float64).reshape(100, 4)
    row_ids = np.asarray([f"row-{idx}" for idx in range(100)], dtype=str)
    split = split_client_partition(
        X=X[partitions_a[0]],
        y=y[partitions_a[0]],
        row_ids=row_ids[partitions_a[0]],
        client_id=0,
        test_size=0.25,
        seed=7,
    )
    assert split.train_size + split.test_size == partitions_a[0].shape[0]
    assert split.row_ids_train.shape[0] == split.train_size
    assert split.row_ids_test.shape[0] == split.test_size

    split_b = split_client_partition(
        X=X[partitions_a[0]],
        y=y[partitions_a[0]],
        row_ids=row_ids[partitions_a[0]],
        client_id=0,
        test_size=0.25,
        seed=7,
    )
    assert split.row_ids_train.tolist() == split_b.row_ids_train.tolist()
    assert split.row_ids_test.tolist() == split_b.row_ids_test.tolist()
