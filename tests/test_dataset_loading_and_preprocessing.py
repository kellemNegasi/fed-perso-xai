from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fed_perso_xai.data.loaders import load_supported_dataset
from fed_perso_xai.data.preprocessing import FrozenTabularPreprocessor
from fed_perso_xai.utils.config import PreprocessingConfig


@pytest.mark.parametrize("dataset_name", ["adult_income", "bank_marketing"])
def test_supported_dataset_loading_smoke(mock_openml, tmp_path, dataset_name: str) -> None:
    frame = mock_openml(dataset_name)
    dataset = load_supported_dataset(dataset_name, cache_dir=tmp_path / "cache")

    assert dataset.X.shape[0] == frame.shape[0]
    assert set(np.unique(dataset.y)) == {0, 1}
    assert dataset.row_ids.shape[0] == dataset.X.shape[0]
    assert dataset.source_metadata["provider"] == "openml"


def test_preprocessor_handles_pathological_columns_and_records_diagnostics(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "age": [30, 40, 35, 50],
            "income": [100.0, 200.0, None, 300.0],
            "job": ["admin", "tech", "admin", None],
            "is_manager": pd.Series([True, False, True, False], dtype="boolean"),
            "constant_flag": ["same", "same", "same", "same"],
            "all_missing": [None, None, None, None],
        }
    )
    preprocessor = FrozenTabularPreprocessor.fit(
        frame,
        PreprocessingConfig(),
        feature_type_overrides={"is_manager": "categorical"},
    )
    transformed_a = preprocessor.transform(frame)
    saved_path = preprocessor.save(tmp_path / "preprocessor.joblib")
    loaded = FrozenTabularPreprocessor.load(saved_path)
    transformed_b, diagnostics = loaded.transform_with_diagnostics(
        frame.assign(job=["admin", "executive", "admin", "executive"]),
        split_name="global_eval",
    )

    assert transformed_a.shape[0] == frame.shape[0]
    assert transformed_a.shape == transformed_b.shape
    metadata = loaded.feature_metadata()
    assert metadata["raw_columns_expected"] == [
        "age",
        "income",
        "job",
        "is_manager",
        "constant_flag",
        "all_missing",
    ]
    assert metadata["drop_reasons"]["constant_columns_removed"] == ["constant_flag"]
    assert metadata["drop_reasons"]["all_missing_columns_removed"] == ["all_missing"]
    assert "income" in metadata["imputed_columns"]
    assert "job" in metadata["imputed_columns"]
    assert "is_manager" in metadata["categorical_columns"]
    assert diagnostics["unknown_categories"]["job"]["values"] == ["executive"]
    assert metadata["encoder_category_vocabularies"]["job"] == ["admin", "tech"]

    lineage = metadata["feature_lineage"]
    job_lineage = [row for row in lineage if row["raw_feature"] == "job"]
    assert all(row["encoded_category"] is not None for row in job_lineage)
    assert metadata["transformed_to_raw_feature_map"][job_lineage[0]["transformed_feature"]] == "job"

    with pytest.raises(ValueError):
        loaded.transform(frame.assign(extra_column=1))
