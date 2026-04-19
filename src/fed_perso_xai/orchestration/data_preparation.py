"""High-level stage-1 data preparation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from fed_perso_xai.data.catalog import DatasetRegistry
from fed_perso_xai.data.loaders import load_supported_dataset
from fed_perso_xai.data.partitioning import (
    ClientSplit,
    dirichlet_partition_labels,
    split_client_partition,
    summarize_labels,
)
from fed_perso_xai.data.preprocessing import FrozenTabularPreprocessor
from fed_perso_xai.data.serialization import (
    ArraySplit,
    SavedDatasetArtifacts,
    SavedPreparedArtifacts,
    save_federated_dataset,
    save_prepared_dataset,
)
from fed_perso_xai.utils.config import DataPreparationConfig
from fed_perso_xai.utils.paths import prepared_dir


@dataclass(frozen=True)
class PreparedDataResult:
    """In-memory summary of a completed preparation run."""

    prepared_artifacts: SavedPreparedArtifacts
    federated_artifacts: SavedDatasetArtifacts
    client_splits: list[ClientSplit]
def prepare_federated_dataset(
    config: DataPreparationConfig,
    *,
    registry: DatasetRegistry | None = None,
) -> PreparedDataResult:
    """Load, preprocess, partition, split, and persist stage-1 prepared data."""

    dataset = load_supported_dataset(
        config.dataset_name,
        cache_dir=config.paths.cache_dir,
        registry=registry,
    )
    raw_train_X, raw_eval_X, raw_train_y, raw_eval_y, raw_train_ids, raw_eval_ids = train_test_split(
        dataset.X,
        dataset.y,
        dataset.row_ids,
        test_size=config.preprocessing.global_eval_size,
        random_state=config.seed,
        stratify=dataset.y,
    )

    preprocessor = FrozenTabularPreprocessor.fit(
        raw_train_X,
        config.preprocessing,
        feature_type_overrides=dataset.spec.feature_type_overrides,
    )
    prepared_root = prepared_dir(config.paths, config.dataset_name, config.seed)
    preprocessor_path = preprocessor.save(prepared_root / "preprocessor.joblib")

    X_train_pool, train_transform_diagnostics = preprocessor.transform_with_diagnostics(
        raw_train_X,
        split_name="global_train",
    )
    X_eval, eval_transform_diagnostics = preprocessor.transform_with_diagnostics(
        raw_eval_X,
        split_name="global_eval",
    )
    y_train_pool = np.asarray(raw_train_y, dtype=np.int64)
    y_eval = np.asarray(raw_eval_y, dtype=np.int64)
    train_row_ids = np.asarray(raw_train_ids, dtype=str)
    eval_row_ids = np.asarray(raw_eval_ids, dtype=str)

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
            row_ids=train_row_ids[indices],
            client_id=client_id,
            test_size=config.preprocessing.client_test_size,
            seed=config.seed,
        )
        for client_id, indices in enumerate(partitions)
    ]
    pooled_client_test = ArraySplit(
        X=np.concatenate([split.X_test for split in client_splits], axis=0),
        y=np.concatenate([split.y_test for split in client_splits], axis=0),
        row_ids=np.concatenate([split.row_ids_test for split in client_splits], axis=0),
    )
    pooled_client_test_metadata = {
        "split_name": "pooled_client_test",
        "num_examples": int(pooled_client_test.y.shape[0]),
        "label_distribution": summarize_labels(pooled_client_test.y),
        "row_id_count": int(pooled_client_test.row_ids.shape[0]),
        "provenance": {
            "source": "pooled_client_test",
            "client_count": config.partition.num_clients,
        },
    }

    prepared_artifacts = save_prepared_dataset(
        config=config,
        dataset_metadata={
            "dataset_name": dataset.name,
            "display_name": dataset.display_name,
            "description": dataset.spec.description,
            "target_column": dataset.spec.target_column,
            "source_metadata": dataset.source_metadata,
            "schema": dataset.schema_summary(),
            "global_train_size": int(y_train_pool.shape[0]),
            "global_eval_size": int(y_eval.shape[0]),
        },
        split_metadata={
            "dataset_name": dataset.name,
            "seed": config.seed,
            "global_train": {
                "split_name": "global_train",
                "num_examples": int(y_train_pool.shape[0]),
                "label_distribution": summarize_labels(y_train_pool),
                "row_id_count": int(train_row_ids.shape[0]),
                "transform_diagnostics": train_transform_diagnostics,
            },
            "global_eval": {
                "split_name": "global_eval",
                "num_examples": int(y_eval.shape[0]),
                "label_distribution": summarize_labels(y_eval),
                "row_id_count": int(eval_row_ids.shape[0]),
                "transform_diagnostics": eval_transform_diagnostics,
            },
            "pooled_client_test": pooled_client_test_metadata,
        },
        feature_metadata=preprocessor.feature_metadata(),
        preprocessor_path=preprocessor_path,
        global_train=ArraySplit(X=X_train_pool, y=y_train_pool, row_ids=train_row_ids),
        global_eval=ArraySplit(X=X_eval, y=y_eval, row_ids=eval_row_ids),
        pooled_client_test=pooled_client_test,
    )
    federated_artifacts = save_federated_dataset(
        dataset_name=config.dataset_name,
        output_root=config.paths.partition_root,
        num_clients=config.partition.num_clients,
        alpha=config.partition.alpha,
        seed=config.seed,
        prepared_root=prepared_artifacts.root_dir,
        preprocessor_path=prepared_artifacts.preprocessor_path,
        feature_metadata_path=prepared_artifacts.feature_metadata_path,
        client_splits=client_splits,
    )

    (prepared_artifacts.root_dir / "partition_metadata.json").write_text(
        json.dumps(
            {
                "num_clients": config.partition.num_clients,
                "alpha": config.partition.alpha,
                "seed": config.seed,
                "partition_root": str(federated_artifacts.root_dir),
                "partition_metadata_path": str(federated_artifacts.partition_metadata_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PreparedDataResult(
        prepared_artifacts=prepared_artifacts,
        federated_artifacts=federated_artifacts,
        client_splits=client_splits,
    )
