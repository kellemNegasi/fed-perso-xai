"""Dataset loading helpers for supported OpenML tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from fed_perso_xai.data.catalog import DEFAULT_DATASET_REGISTRY, DatasetRegistry, DatasetSpec


@dataclass(frozen=True)
class RawTabularDataset:
    """Raw tabular dataset before preprocessing."""

    name: str
    display_name: str
    X: pd.DataFrame
    y: np.ndarray
    row_ids: np.ndarray
    spec: DatasetSpec
    source_metadata: dict[str, object]

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def schema_summary(self) -> dict[str, object]:
        """Return a JSON-friendly summary of the raw schema."""

        return {
            "raw_feature_names": self.feature_names,
            "row_count": int(self.X.shape[0]),
            "feature_count": int(self.X.shape[1]),
            "dtypes": {column: str(dtype) for column, dtype in self.X.dtypes.items()},
            "missing_counts": {
                column: int(count) for column, count in self.X.isna().sum().items()
            },
            "non_null_unique_counts": {
                column: int(self.X[column].dropna().nunique()) for column in self.X.columns
            },
            "feature_type_overrides": self.spec.feature_type_overrides,
            "source_metadata": self.source_metadata,
        }


def load_supported_dataset(
    dataset_name: str,
    cache_dir: Path,
    registry: DatasetRegistry | None = None,
) -> RawTabularDataset:
    """Load a supported dataset from the registry."""

    spec = (registry or DEFAULT_DATASET_REGISTRY).get(dataset_name)
    return load_openml_dataset(spec, cache_dir=cache_dir)


def load_openml_dataset(spec: DatasetSpec, cache_dir: Path) -> RawTabularDataset:
    """Load and normalize one OpenML-backed tabular dataset."""

    bunch = fetch_openml(
        data_id=spec.openml_data_id,
        as_frame=True,
        cache=True,
        data_home=str(cache_dir),
    )
    frame = getattr(bunch, "frame", None)
    target = getattr(bunch, "target", None)
    if frame is not None and isinstance(frame, pd.DataFrame):
        X, y_raw, row_ids = _resolve_frame_target_and_row_ids(frame=frame, target=target, spec=spec)
    else:
        data = getattr(bunch, "data")
        target_values = getattr(bunch, "target")
        feature_names = list(getattr(bunch, "feature_names"))
        X = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=feature_names)
        y_raw = _coerce_target_array(target_values)
        row_ids = np.asarray(X.index.astype(str), dtype=str)

    if spec.cleaning_hook is not None:
        before_shape = X.shape
        X = spec.cleaning_hook(X)
        if X.shape[0] != before_shape[0]:
            raise ValueError(
                f"Dataset cleaning hook for '{spec.key}' changed the number of rows from "
                f"{before_shape[0]} to {X.shape[0]}. Stage-1 cleaning hooks must preserve rows."
            )
    _validate_raw_schema(X, spec)

    y = np.asarray([spec.target_transform(value) for value in y_raw], dtype=np.int64)
    unique_labels = np.unique(y)
    if unique_labels.tolist() != [0, 1]:
        raise ValueError(
            f"Dataset '{spec.key}' did not produce binary labels in {{0,1}}: {unique_labels}."
        )
    return RawTabularDataset(
        name=spec.key,
        display_name=spec.display_name,
        X=X.reset_index(drop=True),
        y=y,
        row_ids=np.asarray(row_ids, dtype=str),
        spec=spec,
        source_metadata={
            "provider": "openml",
            "openml_data_id": int(spec.openml_data_id),
            "openml_name": getattr(bunch, "details", {}).get("name") if hasattr(bunch, "details") else None,
            "openml_version": (
                getattr(bunch, "details", {}).get("version") if hasattr(bunch, "details") else None
            ),
            "cleaning_hook": None if spec.cleaning_hook is None else spec.cleaning_hook.__name__,
        },
    )


def _resolve_frame_target_and_row_ids(
    *,
    frame: pd.DataFrame,
    target: object,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    row_ids = np.asarray(frame.index.astype(str), dtype=str)
    candidate_target_columns = [spec.target_column]
    if hasattr(target, "name"):
        candidate_target_columns.append(getattr(target, "name"))

    for target_column in candidate_target_columns:
        if target_column and target_column in frame.columns:
            X = frame.drop(columns=[target_column]).copy()
            y_raw = frame[target_column].to_numpy(copy=True)
            return X, y_raw, row_ids

    if hasattr(target, "to_numpy"):
        target_array = np.asarray(target.to_numpy(copy=True))
        if target_array.ndim == 1 and target_array.shape[0] == frame.shape[0]:
            return frame.copy(), target_array, row_ids

    fallback_target_name = frame.columns[-1]
    X = frame.iloc[:, :-1].copy()
    return X, frame[fallback_target_name].to_numpy(copy=True), row_ids


def _validate_raw_schema(X: pd.DataFrame, spec: DatasetSpec) -> None:
    if X.columns.duplicated().any():
        duplicate_columns = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate feature columns detected: {duplicate_columns}")
    missing_required = [column for column in spec.required_columns if column not in X.columns]
    if missing_required:
        raise ValueError(
            f"Dataset '{spec.key}' is missing required columns: {missing_required}"
        )


def _coerce_target_array(target: object) -> np.ndarray:
    if target is None:
        raise ValueError("OpenML target is missing and no target column could be resolved.")
    if hasattr(target, "to_numpy"):
        return np.asarray(target.to_numpy(copy=True))
    return np.asarray(target)
