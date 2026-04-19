"""High-level stage-1 data preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from fed_perso_xai.data.loaders import load_supported_dataset
from fed_perso_xai.data.partitioning import (
    ClientSplit,
    dirichlet_partition_labels,
    split_client_partition,
)
from fed_perso_xai.data.preprocessing import FrozenTabularPreprocessor
from fed_perso_xai.data.serialization import SavedDatasetArtifacts, save_federated_dataset
from fed_perso_xai.utils.config import DataPreparationConfig


@dataclass(frozen=True)
class PreparedDataResult:
    """In-memory summary of a completed preparation run."""

    artifacts: SavedDatasetArtifacts
    feature_names: list[str]
    client_splits: list[ClientSplit]


def prepare_federated_dataset(config: DataPreparationConfig) -> PreparedDataResult:
    """Load, preprocess, partition, split, and persist a supported dataset."""

    dataset = load_supported_dataset(config.dataset_name, cache_dir=config.cache_dir)
    raw_train_X, raw_eval_X, raw_train_y, raw_eval_y = train_test_split(
        dataset.X,
        dataset.y,
        test_size=config.preprocessing.global_eval_size,
        random_state=config.seed,
        stratify=dataset.y,
    )

    preprocessor = FrozenTabularPreprocessor.fit(raw_train_X, config.preprocessing)
    X_train_pool = preprocessor.transform(raw_train_X)
    X_eval = preprocessor.transform(raw_eval_X)
    y_train_pool = np.asarray(raw_train_y, dtype=np.int64)
    y_eval = np.asarray(raw_eval_y, dtype=np.int64)

    partitions = dirichlet_partition_labels(
        y=y_train_pool,
        num_clients=config.partition.num_clients,
        alpha=config.partition.alpha,
        seed=config.seed,
        min_client_samples=config.partition.min_client_samples,
        max_retries=config.partition.max_retries,
    )
    client_splits = [
        split_client_partition(
            X=X_train_pool[indices],
            y=y_train_pool[indices],
            client_id=client_id,
            test_size=config.preprocessing.client_test_size,
            seed=config.seed,
        )
        for client_id, indices in enumerate(partitions)
    ]

    artifacts = save_federated_dataset(
        dataset_name=config.dataset_name,
        output_root=config.output_root,
        num_clients=config.partition.num_clients,
        alpha=config.partition.alpha,
        seed=config.seed,
        feature_names=preprocessor.feature_names,
        preprocessing_info=preprocessor.schema_info(),
        client_splits=client_splits,
        global_eval=(X_eval, y_eval),
    )
    return PreparedDataResult(
        artifacts=artifacts,
        feature_names=preprocessor.feature_names,
        client_splits=client_splits,
    )
