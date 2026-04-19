"""Frozen preprocessing for tabular federated learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fed_perso_xai.utils.config import PreprocessingConfig


@dataclass
class FrozenTabularPreprocessor:
    """Fitted preprocessing schema shared by all clients."""

    transformer: ColumnTransformer
    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_names: list[str]
    config: PreprocessingConfig

    @classmethod
    def fit(
        cls,
        X_train: pd.DataFrame,
        config: PreprocessingConfig,
    ) -> "FrozenTabularPreprocessor":
        """Fit the preprocessing schema on the global raw-data training pool."""

        numeric_columns = list(X_train.select_dtypes(include=["number", "bool"]).columns)
        categorical_columns = [
            column for column in X_train.columns if column not in numeric_columns
        ]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=config.numeric_imputation_strategy)),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
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
        )

        transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
        )
        transformer.fit(X_train)
        feature_names = list(transformer.get_feature_names_out())
        return cls(
            transformer=transformer,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            feature_names=feature_names,
            config=config,
        )

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform a dataframe into a dense numeric array."""

        transformed = self.transformer.transform(X)
        return np.asarray(transformed, dtype=np.float64)

    def schema_info(self) -> dict[str, Any]:
        """Return a JSON-friendly preprocessing summary."""

        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "feature_names": self.feature_names,
            "numeric_imputation_strategy": self.config.numeric_imputation_strategy,
            "categorical_imputation_strategy": self.config.categorical_imputation_strategy,
        }
