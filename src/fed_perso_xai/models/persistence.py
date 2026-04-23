"""Persistence helpers for frozen global model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fed_perso_xai.models.base import TabularClassifier
from fed_perso_xai.models.registry import build_model_config, create_model
from fed_perso_xai.utils.paths import federated_model_path, federated_model_metadata_path


@dataclass(frozen=True)
class LoadedGlobalModelArtifact:
    """Reconstructed model plus its persisted metadata."""

    model: TabularClassifier
    metadata: dict[str, Any]
    model_path: Path
    metadata_path: Path


def save_global_model_parameters(path: Path, model: TabularClassifier) -> Path:
    """Persist model parameters in a model-agnostic array bundle."""

    parameters = model.get_parameters()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "parameter_count": np.asarray([len(parameters)], dtype=np.int64),
    }
    for index, parameter in enumerate(parameters):
        payload[f"parameter_{index}"] = np.asarray(parameter, dtype=np.float64)
    np.savez_compressed(path, **payload)
    return path


def load_global_model_parameters(path: Path) -> list[np.ndarray]:
    """Load a persisted parameter bundle."""

    data = np.load(path, allow_pickle=False)
    parameter_count = int(np.asarray(data["parameter_count"], dtype=np.int64).reshape(-1)[0])
    return [
        np.asarray(data[f"parameter_{index}"], dtype=np.float64)
        for index in range(parameter_count)
    ]


def load_global_model(run_dir: Path) -> LoadedGlobalModelArtifact:
    """Load a frozen global model artifact from a completed federated run."""

    metadata_path = federated_model_metadata_path(run_dir)
    model_path = federated_model_path(run_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model metadata at '{metadata_path}'.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact at '{model_path}'.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = str(metadata["model_type"])
    model_config = build_model_config(model_name, dict(metadata.get("model_config") or {}))
    model = create_model(
        model_name,
        n_features=int(metadata["n_features"]),
        config=model_config,
    )
    model.set_parameters(load_global_model_parameters(model_path))
    return LoadedGlobalModelArtifact(
        model=model,
        metadata=metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )
