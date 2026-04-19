from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from fed_perso_xai.data.loaders import load_supported_dataset
from fed_perso_xai.data.preprocessing import FrozenTabularPreprocessor
from fed_perso_xai.utils.config import PreprocessingConfig


@pytest.mark.parametrize(
    ("dataset_name", "target_column", "positive_label"),
    [
        ("adult_income", "class", ">50K"),
        ("bank_marketing", "y", "yes"),
    ],
)
def test_supported_openml_loaders_and_preprocessing_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    dataset_name: str,
    target_column: str,
    positive_label: str,
) -> None:
    frame = pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 28, 37],
            "hours_per_week": [40, 13, 40, 40, 40, 35],
            "workclass": [
                "State-gov",
                "Self-emp-not-inc",
                "Private",
                "Private",
                "Private",
                "Private",
            ],
            target_column: [positive_label, "<=50K", positive_label, "<=50K", positive_label, "<=50K"]
            if dataset_name == "adult_income"
            else ["yes", "no", "yes", "no", "yes", "no"],
        }
    )

    def fake_fetch_openml(*args, **kwargs):
        return SimpleNamespace(frame=frame, target=frame[target_column].rename(target_column))

    monkeypatch.setattr("fed_perso_xai.data.loaders.fetch_openml", fake_fetch_openml)

    dataset = load_supported_dataset(dataset_name, cache_dir=tmp_path / "cache")
    assert dataset.X.shape == (6, 3)
    assert set(np.unique(dataset.y)) == {0, 1}

    raw_train_X, _, raw_train_y, _ = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.33,
        random_state=7,
        stratify=dataset.y,
    )
    preprocessor = FrozenTabularPreprocessor.fit(raw_train_X, PreprocessingConfig())
    X_transformed = preprocessor.transform(raw_train_X)
    assert X_transformed.ndim == 2
    assert X_transformed.shape[0] == raw_train_X.shape[0]
    assert X_transformed.shape[1] == len(preprocessor.feature_names)
    assert raw_train_y.shape[0] == raw_train_X.shape[0]


def test_adult_loader_handles_missing_openml_target_vector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        {
            "V1": [39, 50, 38, 53],
            "V2": ["State-gov", "Self-emp-not-inc", "Private", "Private"],
            "V42": [">50K", "<=50K", ">50K", "<=50K"],
        }
    )

    def fake_fetch_openml(*args, **kwargs):
        return SimpleNamespace(frame=frame, target=None)

    monkeypatch.setattr("fed_perso_xai.data.loaders.fetch_openml", fake_fetch_openml)

    dataset = load_supported_dataset("adult_income", cache_dir=tmp_path / "cache")
    assert list(dataset.X.columns) == ["V1", "V2"]
    assert dataset.y.tolist() == [1, 0, 1, 0]
