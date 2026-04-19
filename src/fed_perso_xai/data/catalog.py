"""Dataset catalog and target normalization adapted from perso-xai."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


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


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative OpenML dataset configuration."""

    key: str
    display_name: str
    openml_data_id: int
    target_transform: Callable[[object], int]
    target_column: str | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "adult_income": DatasetSpec(
        key="adult_income",
        display_name="Adult Income",
        openml_data_id=4535,
        target_transform=_adult_income_transform,
        target_column="V42",
    ),
    "bank_marketing": DatasetSpec(
        key="bank_marketing",
        display_name="Bank Marketing",
        openml_data_id=1461,
        target_transform=_bank_marketing_transform,
        target_column="Class",
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    """Return the dataset specification for a supported dataset key."""

    try:
        return DATASET_SPECS[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}."
        ) from exc
