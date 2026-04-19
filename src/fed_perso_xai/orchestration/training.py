"""High-level stage-1 federated training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fed_perso_xai.data.serialization import load_client_datasets
from fed_perso_xai.fl.client import ClientData
from fed_perso_xai.fl.simulation import SimulationArtifacts, run_federated_training
from fed_perso_xai.utils.config import FederatedTrainingConfig
from fed_perso_xai.utils.paths import partition_root, training_run_dir


def train_from_saved_partitions(
    config: FederatedTrainingConfig,
) -> tuple[SimulationArtifacts, dict[str, Any]]:
    """Load saved client arrays, run federated training, and persist outputs."""

    data_root = partition_root(config.data_root, config.num_clients, config.alpha)
    metadata_path = data_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing prepared dataset metadata at '{metadata_path}'. "
            "Run the prepare-data command first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["dataset_name"] != config.dataset_name:
        raise ValueError(
            f"Prepared dataset name '{metadata['dataset_name']}' does not match "
            f"requested dataset '{config.dataset_name}'."
        )

    client_datasets = [
        ClientData(
            client_id=client.client_id,
            X_train=client.X_train,
            y_train=client.y_train,
            X_test=client.X_test,
            y_test=client.y_test,
        )
        for client in load_client_datasets(data_root, config.num_clients)
    ]
    result_dir = training_run_dir(
        config.results_root,
        config.dataset_name,
        config.num_clients,
        config.alpha,
        config.seed,
    )
    return run_federated_training(
        client_datasets=client_datasets,
        config=config,
        result_dir=result_dir,
    )
