from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fed_perso_xai.data.catalog import DatasetSpec
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
    constant_record = metadata["preprocessing_diagnostics"]["dropped_columns_by_reason"]["constant"][0]
    all_missing_record = metadata["preprocessing_diagnostics"]["dropped_columns_by_reason"]["all_missing"][0]
    assert constant_record["raw_feature"] == "constant_flag"
    assert constant_record["reason"] == "constant"
    assert constant_record["missing_count"] == 0
    assert constant_record["non_null_unique_count"] == 1
    assert all_missing_record["raw_feature"] == "all_missing"
    assert all_missing_record["reason"] == "all_missing"
    assert all_missing_record["missing_count"] == 4
    assert all_missing_record["non_null_unique_count"] == 0
    assert "income" in metadata["imputed_columns"]
    assert "job" in metadata["imputed_columns"]
    assert "is_manager" in metadata["categorical_columns"]
    assert metadata["stable_transformed_feature_order"] == metadata["transformed_feature_names"]
    assert diagnostics["unknown_categories"]["job"]["values"] == ["executive"]
    assert metadata["encoder_category_vocabularies"]["job"] == ["admin", "tech"]

    lineage = metadata["feature_lineage"]
    job_lineage = [row for row in lineage if row["raw_feature"] == "job"]
    assert all(row["encoded_category"] is not None for row in job_lineage)
    assert metadata["transformed_to_raw_feature_map"][job_lineage[0]["transformed_feature"]] == "job"

    with pytest.raises(ValueError):
        loaded.transform(frame.assign(extra_column=1))


def test_preprocessor_drops_constant_columns_before_transformer_fit() -> None:
    frame = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "city": ["a", "b", "a", "b"],
            "constant_numeric": [5, 5, 5, 5],
        }
    )
    preprocessor = FrozenTabularPreprocessor.fit(frame, PreprocessingConfig())
    metadata = preprocessor.feature_metadata()

    assert "constant_numeric" not in preprocessor.kept_raw_feature_names
    assert metadata["drop_reasons"]["constant_columns_removed"] == ["constant_numeric"]
    assert metadata["preprocessing_diagnostics"]["kept_raw_feature_order"] == ["age", "city"]


def test_preprocessor_drops_all_missing_columns_before_transformer_fit() -> None:
    frame = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "segment": ["a", "b", "a", "b"],
            "all_missing_code": [None, None, None, None],
        }
    )
    preprocessor = FrozenTabularPreprocessor.fit(frame, PreprocessingConfig())
    metadata = preprocessor.feature_metadata()

    assert "all_missing_code" not in preprocessor.kept_raw_feature_names
    assert metadata["drop_reasons"]["all_missing_columns_removed"] == ["all_missing_code"]
    assert metadata["preprocessing_diagnostics"]["transformed_feature_order"] == metadata[
        "transformed_feature_names"
    ]


def test_dataset_loading_requires_explicit_target_resolution(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 11, 12, 13],
            "wrong_last_column": [0, 1, 0, 1],
        }
    )
    spec = DatasetSpec(
        key="misconfigured_dataset",
        display_name="Misconfigured Dataset",
        openml_data_id=999,
        target_transform=lambda value: int(value),
        target_column="missing_target",
    )

    def fake_fetch_openml(*args, **kwargs):
        return type("Bunch", (), {"frame": frame, "target": None})()

    monkeypatch.setattr("fed_perso_xai.data.loaders.fetch_openml", fake_fetch_openml)

    with pytest.raises(ValueError, match="Could not resolve the target column"):
        from fed_perso_xai.data.loaders import load_openml_dataset

        load_openml_dataset(spec, cache_dir=tmp_path / "cache")


def test_dataset_loading_requests_configured_openml_target_column(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 11, 12, 13],
            "class": [">50K", "<=50K", ">50K", "<=50K"],
        },
        index=[f"row-{idx}" for idx in range(4)],
    )
    captured_kwargs: dict[str, object] = {}
    spec = DatasetSpec(
        key="adult_income_like",
        display_name="Adult Income Like",
        openml_data_id=999,
        target_transform=lambda value: int(str(value).strip() == ">50K"),
        target_column="class",
    )

    def fake_fetch_openml(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return type(
            "Bunch",
            (),
            {
                "frame": frame.drop(columns=["class"]),
                "target": frame["class"].rename("class"),
                "details": {"name": "adult-income-like", "version": "1"},
            },
        )()

    monkeypatch.setattr("fed_perso_xai.data.loaders.fetch_openml", fake_fetch_openml)

    from fed_perso_xai.data.loaders import load_openml_dataset

    dataset = load_openml_dataset(spec, cache_dir=tmp_path / "cache")

    assert captured_kwargs["target_column"] == "class"
    assert dataset.X.columns.tolist() == ["feature_a", "feature_b"]
    assert dataset.y.tolist() == [1, 0, 1, 0]


def test_dataset_loading_uses_openml_default_target_when_target_column_is_unspecified(
    monkeypatch, tmp_path
) -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 11, 12, 13],
        },
        index=[f"row-{idx}" for idx in range(4)],
    )
    captured_kwargs: dict[str, object] = {}
    spec = DatasetSpec(
        key="bank_marketing_like",
        display_name="Bank Marketing Like",
        openml_data_id=999,
        target_transform=lambda value: int(str(value).strip().lower() == "yes"),
    )

    def fake_fetch_openml(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return type(
            "Bunch",
            (),
            {
                "frame": frame,
                "target": pd.Series(["yes", "no", "yes", "no"], name="Class", index=frame.index),
                "details": {"name": "bank-marketing-like", "version": "1"},
            },
        )()

    monkeypatch.setattr("fed_perso_xai.data.loaders.fetch_openml", fake_fetch_openml)

    from fed_perso_xai.data.loaders import load_openml_dataset

    dataset = load_openml_dataset(spec, cache_dir=tmp_path / "cache")

    assert captured_kwargs["target_column"] == "default-target"
    assert dataset.X.columns.tolist() == ["feature_a", "feature_b"]
    assert dataset.y.tolist() == [1, 0, 1, 0]
