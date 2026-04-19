"""Prepared-data and client-partition persistence helpers."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fed_perso_xai.data.partitioning import ClientSplit, summarize_labels
from fed_perso_xai.utils.config import DataPreparationConfig
from fed_perso_xai.utils.paths import client_dir, partition_root, prepared_dir


@dataclass(frozen=True)
class SavedPreparedArtifacts:
    """Paths to the prepared-data artifacts shared across experiments."""

    root_dir: Path
    config_path: Path
    dataset_metadata_path: Path
    split_metadata_path: Path
    feature_metadata_path: Path
    preprocessor_path: Path
    global_train_path: Path
    global_eval_path: Path
    pooled_client_test_path: Path | None


@dataclass(frozen=True)
class SavedDatasetArtifacts:
    """Pointers to saved client-partition artifacts."""

    root_dir: Path
    partition_metadata_path: Path


@dataclass(frozen=True)
class ArraySplit:
    """Loaded array split with optional row identifiers."""

    X: np.ndarray
    y: np.ndarray
    row_ids: np.ndarray


@dataclass(frozen=True)
class ClientDiskDataset:
    """Loaded arrays for one saved client."""

    client_id: int
    train: ArraySplit
    test: ArraySplit


def save_prepared_dataset(
    *,
    config: DataPreparationConfig,
    dataset_metadata: dict[str, Any],
    split_metadata: dict[str, Any],
    feature_metadata: dict[str, Any],
    preprocessor_path: Path,
    global_train: ArraySplit,
    global_eval: ArraySplit,
    pooled_client_test: ArraySplit | None = None,
) -> SavedPreparedArtifacts:
    """Persist prepared centralized-ready artifacts."""

    root_dir = prepared_dir(config.paths, config.dataset_name, config.seed)
    root_dir.mkdir(parents=True, exist_ok=True)

    config_path = root_dir / "prepare_config.json"
    dataset_metadata_path = root_dir / "dataset_metadata.json"
    split_metadata_path = root_dir / "split_metadata.json"
    feature_metadata_path = root_dir / "feature_metadata.json"
    target_preprocessor_path = root_dir / "preprocessor.joblib"
    global_train_path = root_dir / "global_train.npz"
    global_eval_path = root_dir / "global_eval.npz"

    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    dataset_metadata_path.write_text(json.dumps(dataset_metadata, indent=2), encoding="utf-8")
    split_metadata_path.write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    feature_metadata_path.write_text(json.dumps(feature_metadata, indent=2), encoding="utf-8")
    if preprocessor_path.resolve() != target_preprocessor_path.resolve():
        shutil.copy2(preprocessor_path, target_preprocessor_path)
    _save_array_split(global_train_path, global_train)
    _save_array_split(global_eval_path, global_eval)

    pooled_client_test_path: Path | None = None
    if pooled_client_test is not None:
        pooled_client_test_path = root_dir / "pooled_client_test.npz"
        _save_array_split(pooled_client_test_path, pooled_client_test)

    return SavedPreparedArtifacts(
        root_dir=root_dir,
        config_path=config_path,
        dataset_metadata_path=dataset_metadata_path,
        split_metadata_path=split_metadata_path,
        feature_metadata_path=feature_metadata_path,
        preprocessor_path=target_preprocessor_path,
        global_train_path=global_train_path,
        global_eval_path=global_eval_path,
        pooled_client_test_path=pooled_client_test_path,
    )


def save_federated_dataset(
    *,
    dataset_name: str,
    output_root: Path,
    num_clients: int,
    alpha: float,
    seed: int,
    prepared_root: Path,
    preprocessor_path: Path,
    feature_metadata_path: Path,
    client_splits: list[ClientSplit],
) -> SavedDatasetArtifacts:
    """Persist all client datasets and partition metadata under the required layout."""

    root_dir = partition_root(output_root, num_clients, alpha)
    root_dir.mkdir(parents=True, exist_ok=True)

    for split in client_splits:
        output_dir = client_dir(output_root, num_clients, alpha, split.client_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_array_split(
            output_dir / "train.npz",
            ArraySplit(X=split.X_train, y=split.y_train, row_ids=split.row_ids_train),
        )
        _save_array_split(
            output_dir / "test.npz",
            ArraySplit(X=split.X_test, y=split.y_test, row_ids=split.row_ids_test),
        )

    partition_metadata_path = root_dir / "partition_metadata.json"
    metadata = {
        "dataset_name": dataset_name,
        "num_clients": num_clients,
        "alpha": alpha,
        "seed": seed,
        "prepared_root": str(prepared_root),
        "preprocessor_path": str(preprocessor_path),
        "feature_metadata_path": str(feature_metadata_path),
        "clients": [
            {
                "client_id": split.client_id,
                "train_size": split.train_size,
                "test_size": split.test_size,
                "train_class_distribution": summarize_labels(split.y_train),
                "test_class_distribution": summarize_labels(split.y_test),
            }
            for split in client_splits
        ],
    }
    partition_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (root_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return SavedDatasetArtifacts(root_dir=root_dir, partition_metadata_path=partition_metadata_path)


def load_array_split(path: Path) -> ArraySplit:
    """Load an array split artifact."""

    data = np.load(path, allow_pickle=False)
    row_ids = np.asarray(data["row_ids"], dtype=str) if "row_ids" in data else np.asarray([], dtype=str)
    return ArraySplit(
        X=np.asarray(data["X"], dtype=np.float64),
        y=np.asarray(data["y"], dtype=np.int64),
        row_ids=row_ids,
    )


def load_client_datasets(root_dir: Path, num_clients: int) -> list[ClientDiskDataset]:
    """Load all saved client partitions from disk."""

    return [
        ClientDiskDataset(
            client_id=client_id,
            train=load_array_split(root_dir / f"client_{client_id}" / "train.npz"),
            test=load_array_split(root_dir / f"client_{client_id}" / "test.npz"),
        )
        for client_id in range(num_clients)
    ]


def copy_shared_artifacts(prepared_root: Path, destination_dir: Path) -> None:
    """Copy the fitted preprocessor and feature metadata into a result directory."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "preprocessor.joblib",
        "feature_metadata.json",
        "dataset_metadata.json",
        "split_metadata.json",
    ):
        source = prepared_root / filename
        if source.exists():
            shutil.copy2(source, destination_dir / filename)


def _save_array_split(path: Path, split: ArraySplit) -> None:
    np.savez_compressed(
        path,
        X=split.X.astype(np.float32, copy=False),
        y=split.y.astype(np.int64, copy=False),
        row_ids=np.asarray(split.row_ids, dtype=str),
    )
