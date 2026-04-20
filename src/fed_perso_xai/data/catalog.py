"""Dataset registry and dataset-specific hooks for tabular stage-1 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


DatasetCleaningHook = Callable[[pd.DataFrame], pd.DataFrame]
TargetTransform = Callable[[object], int]


def _normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    return text.replace(".", "").replace(" ", "")


def _adult_income_transform(value: object) -> int:
    positive_tokens = {
        "50000+",
        ">50k",
        "50k+",
        ">50000",
        "morethan50k",
        "morethan50000",
    }
    return int(_normalize_text(value) in positive_tokens)


def _bank_marketing_transform(value: object) -> int:
    return int(_normalize_text(value) in {"yes", "1", "true"})


def _replace_common_missing_tokens(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize textual missing-value markers before generic preprocessing."""

    cleaned = frame.copy()
    object_columns = cleaned.select_dtypes(include=["object", "category", "string"]).columns
    if len(object_columns) > 0:
        cleaned[object_columns] = cleaned[object_columns].replace({"?": pd.NA, "unknown": "unknown"})
    return cleaned


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative dataset specification used by the registry."""

    key: str
    display_name: str
    openml_data_id: int
    target_transform: TargetTransform
    target_column: str | None = None
    cleaning_hook: DatasetCleaningHook | None = None
    feature_type_overrides: dict[str, str] = field(default_factory=dict)
    required_columns: tuple[str, ...] = ()
    description: str = ""


class DatasetRegistry:
    """Registry of supported dataset specifications."""

    def __init__(self, specs: list[DatasetSpec] | None = None) -> None:
        self._specs: dict[str, DatasetSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: DatasetSpec) -> None:
        if spec.key in self._specs:
            raise ValueError(f"Dataset '{spec.key}' is already registered.")
        self._specs[spec.key] = spec

    def get(self, key: str) -> DatasetSpec:
        try:
            return self._specs[key]
        except KeyError as exc:
            supported = ", ".join(sorted(self._specs))
            raise ValueError(
                f"Unsupported dataset '{key}'. Supported datasets: {supported}."
            ) from exc

    def list_keys(self) -> list[str]:
        return sorted(self._specs)


DEFAULT_DATASET_REGISTRY = DatasetRegistry(
    specs=[
        DatasetSpec(
            key="adult_income",
            display_name="Adult Income",
            openml_data_id=1590,
            target_transform=_adult_income_transform,
            target_column="class",
            cleaning_hook=_replace_common_missing_tokens,
            description="OpenML Adult Income binary classification benchmark.",
        ),
        DatasetSpec(
            key="bank_marketing",
            display_name="Bank Marketing",
            openml_data_id=1461,
            target_transform=_bank_marketing_transform,
            cleaning_hook=_replace_common_missing_tokens,
            description="OpenML Bank Marketing binary classification benchmark.",
        ),
    ]
)


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    """Return the dataset specification for a supported dataset key."""

    return DEFAULT_DATASET_REGISTRY.get(dataset_name)
