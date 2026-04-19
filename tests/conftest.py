from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _build_mock_frame(dataset_name: str) -> tuple[pd.DataFrame, str]:
    if dataset_name == "adult_income":
        target_column = "class"
        frame = pd.DataFrame(
            {
                "age": [39, 50, 38, 53, 28, 37, 49, 52, 31, 42, 44, 29],
                "hours_per_week": [40, 13, 40, 40, 40, 35, 45, 20, 50, 38, 42, 60],
                "education_num": [13, 13, 9, 7, 13, 14, 10, 12, 9, 11, 10, 13],
                "workclass": [
                    "State-gov",
                    "Self-emp-not-inc",
                    "Private",
                    "Private",
                    "?",
                    "Private",
                    "Private",
                    "State-gov",
                    "Private",
                    "Private",
                    "Local-gov",
                    "Private",
                ],
                "marital_status": [
                    "Never-married",
                    "Married-civ-spouse",
                    "Divorced",
                    "Married-civ-spouse",
                    "Married-civ-spouse",
                    "Married-civ-spouse",
                    "Divorced",
                    "Never-married",
                    "Separated",
                    "Married-civ-spouse",
                    "Never-married",
                    "Married-civ-spouse",
                ],
                "constant_flag": ["same"] * 12,
                target_column: [
                    ">50K",
                    "<=50K",
                    ">50K",
                    "<=50K",
                    ">50K",
                    "<=50K",
                    ">50K",
                    "<=50K",
                    ">50K",
                    "<=50K",
                    ">50K",
                    "<=50K",
                ],
            },
            index=[f"adult-{idx}" for idx in range(12)],
        )
        return frame, target_column

    target_column = "y"
    frame = pd.DataFrame(
        {
            "age": [30, 41, 52, 36, 45, 27, 50, 29, 33, 48, 39, 55],
            "balance": [100, 250, -10, 400, 120, 60, 310, -30, 45, 500, 210, 150],
            "job": [
                "admin.",
                "technician",
                "services",
                "management",
                "blue-collar",
                "?",
                "retired",
                "student",
                "admin.",
                "management",
                "technician",
                "retired",
            ],
            "marital": [
                "single",
                "married",
                "divorced",
                "married",
                "single",
                "single",
                "married",
                "single",
                "divorced",
                "married",
                "single",
                "married",
            ],
            "constant_flag": ["same"] * 12,
            target_column: ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        },
        index=[f"bank-{idx}" for idx in range(12)],
    )
    return frame, target_column


@pytest.fixture
def mock_openml(monkeypatch: pytest.MonkeyPatch):
    def _factory(dataset_name: str):
        frame, target_column = _build_mock_frame(dataset_name)

        def fake_fetch_openml(*args, **kwargs):
            return SimpleNamespace(
                frame=frame,
                target=frame[target_column].rename(target_column),
            )

        monkeypatch.setattr("fed_perso_xai.data.loaders.fetch_openml", fake_fetch_openml)
        return frame

    return _factory


@pytest.fixture
def synthetic_client_splits():
    rng = np.random.default_rng(0)
    splits = []
    for client_id in range(3):
        X_train = rng.normal(size=(16, 4))
        logits_train = X_train[:, 0] - 0.5 * X_train[:, 1]
        y_train = (logits_train > 0.0).astype(np.int64)
        X_test = rng.normal(size=(8, 4))
        logits_test = X_test[:, 0] - 0.5 * X_test[:, 1]
        y_test = (logits_test > 0.0).astype(np.int64)
        splits.append(
            {
                "client_id": client_id,
                "X_train": X_train,
                "y_train": y_train,
                "row_ids_train": np.asarray([f"train-{client_id}-{idx}" for idx in range(16)], dtype=str),
                "X_test": X_test,
                "y_test": y_test,
                "row_ids_test": np.asarray([f"test-{client_id}-{idx}" for idx in range(8)], dtype=str),
            }
        )
    return splits
