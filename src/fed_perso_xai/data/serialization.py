"""On-disk persistence for processed client datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fed_perso_xai.data.partitioning import ClientSplit, summarize_labels
from fed_perso_xai.utils.paths import client_dir, partition_root


@dataclass(frozen=True)
class SavedDatasetArtifacts:
    """Pointers to saved dataset artifacts."""

    root_dir: Path
    metadata_path: Path
    global_eval_path: Path | None


@dataclass(frozen=True)
class ClientDiskDataset:
    """Loaded arrays for one saved client."""

    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def save_federated_dataset(
    *,
    dataset_name: str,
    output_root: Path,
    num_clients: int,
    alpha: float,
    seed: int,
    feature_names: list[str],
    preprocessing_info: dict[str, Any],
    client_splits: list[ClientSplit],
    global_eval: tuple[np.ndarray, np.ndarray] | None = None,
) -> SavedDatasetArtifacts:
    """Persist all client datasets and metadata under the required layout."""

    root_dir = partition_root(output_root, num_clients, alpha)
    root_dir.mkdir(parents=True, exist_ok=True)

    for split in client_splits:
        output_dir = client_dir(output_root, num_clients, alpha, split.client_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "train.npz",
            X=split.X_train.astype(np.float32, copy=False),
            y=split.y_train.astype(np.int64, copy=False),
        )
        np.savez_compressed(
            output_dir / "test.npz",
            X=split.X_test.astype(np.float32, copy=False),
            y=split.y_test.astype(np.int64, copy=False),
        )

    global_eval_path: Path | None = None
    if global_eval is not None:
        global_eval_path = root_dir / "global_eval.npz"
        np.savez_compressed(
            global_eval_path,
            X=global_eval[0].astype(np.float32, copy=False),
            y=global_eval[1].astype(np.int64, copy=False),
        )

    metadata_path = root_dir / "metadata.json"
    metadata = {
        "dataset_name": dataset_name,
        "num_clients": num_clients,
        "alpha": alpha,
        "seed": seed,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "preprocessing": preprocessing_info,
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
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return SavedDatasetArtifacts(
        root_dir=root_dir,
        metadata_path=metadata_path,
        global_eval_path=global_eval_path,
    )


def load_client_datasets(
    root_dir: Path,
    num_clients: int,
) -> list[ClientDiskDataset]:
    """Load all saved client partitions from disk."""

    clients: list[ClientDiskDataset] = []
    for client_id in range(num_clients):
        train_data = np.load(root_dir / f"client_{client_id}" / "train.npz")
        test_data = np.load(root_dir / f"client_{client_id}" / "test.npz")
        clients.append(
            ClientDiskDataset(
                client_id=client_id,
                X_train=np.asarray(train_data["X"], dtype=np.float64),
                y_train=np.asarray(train_data["y"], dtype=np.int64),
                X_test=np.asarray(test_data["X"], dtype=np.float64),
                y_test=np.asarray(test_data["y"], dtype=np.int64),
            )
        )
    return clients
