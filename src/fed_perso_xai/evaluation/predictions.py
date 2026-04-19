"""Prediction artifact helpers shared by centralized and federated runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PredictionArtifact:
    """One machine-readable prediction artifact."""

    run_id: str
    dataset_name: str
    split_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    row_ids: np.ndarray
    client_ids: np.ndarray

    @property
    def num_examples(self) -> int:
        return int(self.y_true.shape[0])


def build_prediction_artifact(
    *,
    run_id: str,
    dataset_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    row_ids: np.ndarray,
    client_ids: np.ndarray | None = None,
    threshold: float = 0.5,
) -> PredictionArtifact:
    """Build a reusable prediction artifact."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    if client_ids is None:
        client_ids = np.full(y_true.shape[0], -1, dtype=np.int64)
    return PredictionArtifact(
        run_id=run_id,
        dataset_name=dataset_name,
        split_name=split_name,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        row_ids=np.asarray(row_ids, dtype=str),
        client_ids=np.asarray(client_ids, dtype=np.int64),
    )


def save_prediction_artifact(path: Path, artifact: PredictionArtifact) -> Path:
    """Persist one prediction artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        run_id=np.asarray([artifact.run_id], dtype=str),
        dataset_name=np.asarray([artifact.dataset_name], dtype=str),
        split_name=np.asarray([artifact.split_name], dtype=str),
        y_true=artifact.y_true,
        y_pred=artifact.y_pred,
        y_prob=artifact.y_prob,
        row_ids=artifact.row_ids,
        client_ids=artifact.client_ids,
    )
    return path


def load_prediction_artifact(path: Path) -> PredictionArtifact:
    """Load one prediction artifact from disk."""

    data = np.load(path, allow_pickle=False)
    return PredictionArtifact(
        run_id=str(np.asarray(data["run_id"], dtype=str)[0]),
        dataset_name=str(np.asarray(data["dataset_name"], dtype=str)[0]),
        split_name=str(np.asarray(data["split_name"], dtype=str)[0]),
        y_true=np.asarray(data["y_true"], dtype=np.int64),
        y_pred=np.asarray(data["y_pred"], dtype=np.int64),
        y_prob=np.asarray(data["y_prob"], dtype=np.float64),
        row_ids=np.asarray(data["row_ids"], dtype=str),
        client_ids=np.asarray(data["client_ids"], dtype=np.int64),
    )
