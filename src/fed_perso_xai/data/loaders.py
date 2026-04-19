"""Dataset loading helpers for supported OpenML tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from fed_perso_xai.data.catalog import DatasetSpec, get_dataset_spec


@dataclass(frozen=True)
class RawTabularDataset:
    """Raw tabular dataset before preprocessing."""

    name: str
    display_name: str
    X: pd.DataFrame
    y: np.ndarray

    @property
    def feature_names(self) -> list[str]:
        """Return raw feature names."""

        return list(self.X.columns)


def load_supported_dataset(dataset_name: str, cache_dir: Path) -> RawTabularDataset:
    """Load one supported dataset from OpenML into a dataframe + binary labels."""

    spec = get_dataset_spec(dataset_name)
    return load_openml_dataset(spec, cache_dir=cache_dir)


def load_openml_dataset(spec: DatasetSpec, cache_dir: Path) -> RawTabularDataset:
    """Load and normalize a supported OpenML dataset."""

    bunch = fetch_openml(
        data_id=spec.openml_data_id,
        as_frame=True,
        cache=True,
        data_home=str(cache_dir),
    )
    frame = getattr(bunch, "frame", None)
    target = getattr(bunch, "target", None)
    if frame is not None and isinstance(frame, pd.DataFrame):
        X, y_raw = _resolve_frame_and_target(frame=frame, target=target, spec=spec)
    else:
        data = getattr(bunch, "data")
        target_values = getattr(bunch, "target")
        if not isinstance(data, pd.DataFrame):
            X = pd.DataFrame(data, columns=list(getattr(bunch, "feature_names")))
        else:
            X = data.copy()
        y_raw = _coerce_target_array(target_values)

    y = np.asarray([spec.target_transform(value) for value in y_raw], dtype=np.int64)
    unique_labels = np.unique(y)
    if unique_labels.tolist() != [0, 1]:
        raise ValueError(
            f"Dataset '{spec.key}' did not produce binary labels in {{0,1}}: {unique_labels}."
        )
    return RawTabularDataset(
        name=spec.key,
        display_name=spec.display_name,
        X=X,
        y=y,
    )


def _resolve_frame_and_target(
    *,
    frame: pd.DataFrame,
    target: object,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Resolve a target vector from OpenML frame-oriented outputs."""

    if hasattr(target, "to_numpy") and getattr(target, "shape", (0,))[0] == frame.shape[0]:
        target_name = getattr(target, "name", None)
        X = frame.drop(columns=[target_name], errors="ignore").copy()
        return X, target.to_numpy(copy=True)

    if spec.target_column and spec.target_column in frame.columns:
        X = frame.drop(columns=[spec.target_column]).copy()
        return X, frame[spec.target_column].to_numpy(copy=True)

    if target is not None:
        target_array = _coerce_target_array(target)
        if target_array.ndim == 1 and target_array.shape[0] == frame.shape[0]:
            return frame.copy(), target_array
        if target_array.ndim == 0 and str(target_array.item()) in frame.columns:
            target_name = str(target_array.item())
            X = frame.drop(columns=[target_name]).copy()
            return X, frame[target_name].to_numpy(copy=True)

    fallback_target_name = frame.columns[-1]
    X = frame.iloc[:, :-1].copy()
    return X, frame[fallback_target_name].to_numpy(copy=True)


def _coerce_target_array(target: object) -> np.ndarray:
    """Convert an arbitrary OpenML target object into a NumPy array."""

    if target is None:
        raise ValueError("OpenML target is missing and no target column could be resolved.")
    if hasattr(target, "to_numpy"):
        return np.asarray(target.to_numpy(copy=True))
    return np.asarray(target)
