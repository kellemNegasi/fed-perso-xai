"""Frozen preprocessing with schema validation and feature-lineage metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fed_perso_xai.utils.config import PreprocessingConfig


@dataclass(frozen=True)
class ColumnDropRecord:
    """Reason why one raw input column was removed before fitting."""

    raw_feature: str
    reason: str
    dtype: str
    missing_count: int
    non_null_unique_count: int


@dataclass(frozen=True)
class ColumnProfile:
    """Schema diagnostics for one raw input column."""

    raw_feature: str
    dtype: str
    missing_count: int
    non_null_unique_count: int
    inferred_type: str
    kept: bool
    drop_reason: str | None
    had_missing_values: bool
    type_override: str | None


@dataclass(frozen=True)
class FeatureLineageRecord:
    """Mapping from one transformed feature back to the originating raw column."""

    transformed_feature: str
    raw_feature: str
    feature_group: str
    transformer_name: str
    derived_feature_type: str
    encoded_category: str | None = None


@dataclass(frozen=True)
class TransformDiagnostics:
    """Diagnostics emitted each time a frame is transformed."""

    split_name: str
    num_rows: int
    unknown_categories: dict[str, dict[str, Any]]


@dataclass
class FrozenTabularPreprocessor:
    """Fitted preprocessing schema shared by all experiments."""

    transformer: ColumnTransformer
    expected_raw_feature_names: list[str]
    kept_raw_feature_names: list[str]
    dropped_columns: list[ColumnDropRecord]
    numeric_columns: list[str]
    categorical_columns: list[str]
    imputed_columns: list[str]
    feature_names: list[str]
    feature_lineage: list[FeatureLineageRecord]
    encoder_category_vocabularies: dict[str, list[str]]
    column_profiles: list[ColumnProfile]
    feature_type_overrides: dict[str, str]
    config: PreprocessingConfig

    @classmethod
    def fit(
        cls,
        X_train: pd.DataFrame,
        config: PreprocessingConfig,
        *,
        feature_type_overrides: dict[str, str] | None = None,
    ) -> "FrozenTabularPreprocessor":
        """Fit the preprocessing schema on the global raw-data training pool."""

        feature_type_overrides = feature_type_overrides or {}
        expected_raw_feature_names = list(X_train.columns)
        _validate_frame_schema(X_train, expected_columns=expected_raw_feature_names)
        _validate_feature_type_overrides(
            expected_columns=expected_raw_feature_names,
            feature_type_overrides=feature_type_overrides,
        )

        kept_raw_feature_names, dropped_columns, column_profiles, imputed_columns = (
            _profile_and_select_columns(X_train, feature_type_overrides)
        )
        if not kept_raw_feature_names:
            raise ValueError(
                "Preprocessing removed every raw feature column. At least one non-constant, "
                "non-empty column is required."
            )

        kept_frame = _prepare_frame_for_transformer(
            X_train[kept_raw_feature_names],
            categorical_columns=[],
        )
        numeric_columns, categorical_columns = _infer_feature_groups(
            kept_frame,
            feature_type_overrides,
        )
        if not numeric_columns and not categorical_columns:
            raise ValueError("No numeric or categorical columns remain after preprocessing.")

        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy=config.numeric_imputation_strategy)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                )
            )
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy=config.categorical_imputation_strategy),
                            ),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    categorical_columns,
                )
            )

        transformer = ColumnTransformer(transformers=transformers)
        fit_frame = _prepare_frame_for_transformer(
            kept_frame,
            categorical_columns=categorical_columns,
        )
        transformer.fit(fit_frame)
        feature_names = list(transformer.get_feature_names_out())
        encoder_category_vocabularies = _extract_encoder_vocabularies(
            transformer=transformer,
            categorical_columns=categorical_columns,
        )
        feature_lineage = _build_feature_lineage(
            transformer=transformer,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            encoder_category_vocabularies=encoder_category_vocabularies,
        )
        return cls(
            transformer=transformer,
            expected_raw_feature_names=expected_raw_feature_names,
            kept_raw_feature_names=kept_raw_feature_names,
            dropped_columns=dropped_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            imputed_columns=imputed_columns,
            feature_names=feature_names,
            feature_lineage=feature_lineage,
            encoder_category_vocabularies=encoder_category_vocabularies,
            column_profiles=column_profiles,
            feature_type_overrides=dict(feature_type_overrides),
            config=config,
        )

    @property
    def raw_feature_names(self) -> list[str]:
        """Backwards-compatible alias for the expected raw schema."""

        return self.expected_raw_feature_names

    def validate_frame_schema(self, X: pd.DataFrame) -> None:
        """Validate that a frame matches the fitted preprocessing schema."""

        _validate_frame_schema(X, expected_columns=self.expected_raw_feature_names)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform a dataframe into a dense numeric array with stable ordering."""

        transformed, _ = self.transform_with_diagnostics(X, split_name="unspecified")
        return transformed

    def transform_with_diagnostics(
        self,
        X: pd.DataFrame,
        *,
        split_name: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Transform a dataframe and emit schema/unknown-category diagnostics."""

        self.validate_frame_schema(X)
        kept_frame = _prepare_frame_for_transformer(
            X[self.kept_raw_feature_names],
            categorical_columns=self.categorical_columns,
        )
        diagnostics = TransformDiagnostics(
            split_name=split_name,
            num_rows=int(kept_frame.shape[0]),
            unknown_categories=_detect_unknown_categories(self, kept_frame),
        )
        transformed = self.transformer.transform(kept_frame)
        return np.asarray(transformed, dtype=np.float64), {
            "split_name": diagnostics.split_name,
            "num_rows": diagnostics.num_rows,
            "unknown_categories": diagnostics.unknown_categories,
        }

    def save(self, path: Path) -> Path:
        """Persist the fitted preprocessor for later reuse."""

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "FrozenTabularPreprocessor":
        """Load a persisted fitted preprocessor."""

        loaded = joblib.load(path)
        if not isinstance(loaded, FrozenTabularPreprocessor):
            raise TypeError("Loaded object is not a FrozenTabularPreprocessor.")
        return loaded

    def feature_metadata(self) -> dict[str, Any]:
        """Return JSON-friendly metadata for downstream consumers."""

        dropped_columns = [
            {
                "raw_feature": record.raw_feature,
                "reason": record.reason,
                "dtype": record.dtype,
                "missing_count": record.missing_count,
                "non_null_unique_count": record.non_null_unique_count,
            }
            for record in self.dropped_columns
        ]
        transformed_to_raw = {
            record.transformed_feature: record.raw_feature for record in self.feature_lineage
        }
        raw_to_transformed: dict[str, list[str]] = {column: [] for column in self.kept_raw_feature_names}
        for record in self.feature_lineage:
            raw_to_transformed.setdefault(record.raw_feature, []).append(record.transformed_feature)
        dropped_by_reason = {
            "constant": [
                record for record in dropped_columns if record["reason"] == "constant"
            ],
            "all_missing": [
                record for record in dropped_columns if record["reason"] == "all_missing"
            ],
        }

        return {
            "schema_version": "stage1_feature_metadata_v3",
            "raw_columns_expected": self.expected_raw_feature_names,
            "raw_columns_kept": self.kept_raw_feature_names,
            "raw_columns_dropped": [record["raw_feature"] for record in dropped_columns],
            "dropped_columns": dropped_columns,
            "drop_reasons": {
                "constant_columns_removed": [
                    record["raw_feature"] for record in dropped_by_reason["constant"]
                ],
                "all_missing_columns_removed": [
                    record["raw_feature"] for record in dropped_by_reason["all_missing"]
                ],
            },
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "imputed_columns": self.imputed_columns,
            "transformed_feature_names": self.feature_names,
            "stable_transformed_feature_order": self.feature_names,
            "transformed_to_raw_feature_map": transformed_to_raw,
            "raw_to_transformed_feature_map": raw_to_transformed,
            "encoder_category_vocabularies": self.encoder_category_vocabularies,
            "feature_lineage": [
                {
                    "transformed_feature": record.transformed_feature,
                    "raw_feature": record.raw_feature,
                    "feature_group": record.feature_group,
                    "transformer_name": record.transformer_name,
                    "derived_feature_type": record.derived_feature_type,
                    "encoded_category": record.encoded_category,
                }
                for record in self.feature_lineage
            ],
            "feature_type_overrides": self.feature_type_overrides,
            "preprocessing_diagnostics": {
                "dropped_column_summary": {
                    "total_dropped": len(dropped_columns),
                    "constant_columns": [
                        record["raw_feature"] for record in dropped_by_reason["constant"]
                    ],
                    "all_missing_columns": [
                        record["raw_feature"] for record in dropped_by_reason["all_missing"]
                    ],
                },
                "dropped_columns_by_reason": {
                    reason: rows for reason, rows in dropped_by_reason.items() if rows
                },
                "kept_raw_feature_order": self.kept_raw_feature_names,
                "transformed_feature_order": self.feature_names,
            },
            "schema_diagnostics": {
                "raw_column_count": len(self.expected_raw_feature_names),
                "kept_column_count": len(self.kept_raw_feature_names),
                "dropped_column_count": len(self.dropped_columns),
                "column_profiles": [
                    {
                        "raw_feature": profile.raw_feature,
                        "dtype": profile.dtype,
                        "missing_count": profile.missing_count,
                        "non_null_unique_count": profile.non_null_unique_count,
                        "inferred_type": profile.inferred_type,
                        "kept": profile.kept,
                        "drop_reason": profile.drop_reason,
                        "had_missing_values": profile.had_missing_values,
                        "type_override": profile.type_override,
                    }
                    for profile in self.column_profiles
                ],
            },
            "unknown_category_policy": "ignore_and_zero_encode_with_reporting",
            "numeric_imputation_strategy": self.config.numeric_imputation_strategy,
            "categorical_imputation_strategy": self.config.categorical_imputation_strategy,
        }


def _validate_frame_schema(X: pd.DataFrame, expected_columns: list[str]) -> None:
    if X.columns.duplicated().any():
        duplicate_columns = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate feature columns detected: {duplicate_columns}")
    actual_columns = list(X.columns)
    missing_columns = [column for column in expected_columns if column not in actual_columns]
    unexpected_columns = [column for column in actual_columns if column not in expected_columns]
    if missing_columns or unexpected_columns:
        raise ValueError(
            "Frame schema mismatch. "
            f"Missing columns: {missing_columns}. Unexpected columns: {unexpected_columns}."
        )


def _validate_feature_type_overrides(
    *,
    expected_columns: list[str],
    feature_type_overrides: dict[str, str],
) -> None:
    unsupported_columns = [
        column for column in feature_type_overrides if column not in expected_columns
    ]
    if unsupported_columns:
        raise ValueError(
            f"Feature type overrides reference unknown columns: {unsupported_columns}"
        )
    unsupported_types = {
        column: value
        for column, value in feature_type_overrides.items()
        if value not in {"numeric", "categorical"}
    }
    if unsupported_types:
        raise ValueError(
            "Feature type overrides must be 'numeric' or 'categorical'. "
            f"Received: {unsupported_types}"
        )


def _profile_and_select_columns(
    X: pd.DataFrame,
    feature_type_overrides: dict[str, str],
) -> tuple[list[str], list[ColumnDropRecord], list[ColumnProfile], list[str]]:
    kept_columns: list[str] = []
    dropped_columns: list[ColumnDropRecord] = []
    column_profiles: list[ColumnProfile] = []
    imputed_columns: list[str] = []

    for column in X.columns:
        series = X[column]
        missing_count = int(series.isna().sum())
        non_null_unique_count = int(series.dropna().nunique())
        override = feature_type_overrides.get(column)
        inferred_type = override or _infer_column_type(series)
        drop_reason: str | None = None

        if missing_count == int(series.shape[0]):
            drop_reason = "all_missing"
        elif non_null_unique_count <= 1:
            drop_reason = "constant"

        if missing_count > 0 and drop_reason is None:
            imputed_columns.append(column)

        kept = drop_reason is None
        column_profiles.append(
            ColumnProfile(
                raw_feature=column,
                dtype=str(series.dtype),
                missing_count=missing_count,
                non_null_unique_count=non_null_unique_count,
                inferred_type=inferred_type,
                kept=kept,
                drop_reason=drop_reason,
                had_missing_values=missing_count > 0,
                type_override=override,
            )
        )
        if kept:
            kept_columns.append(column)
        else:
            dropped_columns.append(
                ColumnDropRecord(
                    raw_feature=column,
                    reason=drop_reason or "unknown",
                    dtype=str(series.dtype),
                    missing_count=missing_count,
                    non_null_unique_count=non_null_unique_count,
                )
            )

    return kept_columns, dropped_columns, column_profiles, imputed_columns


def _infer_feature_groups(
    X: pd.DataFrame,
    feature_type_overrides: dict[str, str],
) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in X.columns:
        override = feature_type_overrides.get(column)
        if override == "numeric":
            numeric_columns.append(column)
            continue
        if override == "categorical":
            categorical_columns.append(column)
            continue
        inferred_type = _infer_column_type(X[column])
        if inferred_type == "numeric":
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _extract_encoder_vocabularies(
    *,
    transformer: ColumnTransformer,
    categorical_columns: list[str],
) -> dict[str, list[str]]:
    if not categorical_columns:
        return {}
    encoder = transformer.named_transformers_["categorical"].named_steps["encoder"]
    return {
        column: [_normalize_category_value(category) for category in categories]
        for column, categories in zip(categorical_columns, encoder.categories_, strict=False)
    }


def _build_feature_lineage(
    *,
    transformer: ColumnTransformer,
    numeric_columns: list[str],
    categorical_columns: list[str],
    encoder_category_vocabularies: dict[str, list[str]],
) -> list[FeatureLineageRecord]:
    lineage: list[FeatureLineageRecord] = []
    for column in numeric_columns:
        lineage.append(
            FeatureLineageRecord(
                transformed_feature=f"numeric__{column}",
                raw_feature=column,
                feature_group="numeric",
                transformer_name="standard_scaler",
                derived_feature_type="scaled_numeric",
            )
        )

    if categorical_columns:
        encoder = transformer.named_transformers_["categorical"].named_steps["encoder"]
        encoded_names = list(encoder.get_feature_names_out(categorical_columns))
        encoded_categories = [
            (column, category)
            for column in categorical_columns
            for category in encoder_category_vocabularies[column]
        ]
        for encoded_name, (column, category) in zip(encoded_names, encoded_categories, strict=False):
            lineage.append(
                FeatureLineageRecord(
                    transformed_feature=f"categorical__{encoded_name}",
                    raw_feature=column,
                    feature_group="categorical",
                    transformer_name="one_hot_encoder",
                    derived_feature_type="one_hot_category",
                    encoded_category=category,
                )
            )
    return lineage


def _detect_unknown_categories(
    preprocessor: FrozenTabularPreprocessor,
    X: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    if not preprocessor.categorical_columns:
        return {}

    categorical_pipeline = preprocessor.transformer.named_transformers_["categorical"]
    imputer = categorical_pipeline.named_steps["imputer"]
    unknowns: dict[str, dict[str, Any]] = {}

    for column, statistic in zip(
        preprocessor.categorical_columns,
        imputer.statistics_,
        strict=False,
    ):
        series = X[column]
        filled = series.fillna(statistic)
        known_values = set(preprocessor.encoder_category_vocabularies[column])
        unseen_values = [
            _normalize_category_value(value)
            for value in pd.unique(filled)
            if _normalize_category_value(value) not in known_values
        ]
        if unseen_values:
            counts = {
                value: int((filled.astype(str) == value).sum()) for value in unseen_values
            }
            unknowns[column] = {
                "count": int(sum(counts.values())),
                "values": unseen_values,
                "counts": counts,
            }
    return unknowns


def _normalize_category_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _prepare_frame_for_transformer(
    X: pd.DataFrame,
    *,
    categorical_columns: list[str],
) -> pd.DataFrame:
    prepared = X.copy()
    for column in categorical_columns:
        prepared[column] = prepared[column].astype(object)
    return prepared
