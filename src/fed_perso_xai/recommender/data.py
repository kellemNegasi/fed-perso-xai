"""Build pairwise recommender training arrays from client-local artifacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

STATISTICAL_DATASET_FEATURES = {
    "dataset_log_dataset_size_z",
    "dataset_mean_of_means_z",
    "dataset_std_of_means_z",
    "dataset_mean_variance_z",
    "dataset_max_variance_z",
    "dataset_mean_skewness_z",
    "dataset_std_skewness_z",
    "dataset_max_kurtosis_z",
    "dataset_mean_std_z",
    "dataset_std_std_z",
    "dataset_max_std_z",
    "dataset_mean_range_z",
    "dataset_max_range_z",
    "dataset_mean_cardinality_z",
    "dataset_max_cardinality_z",
    "dataset_mean_cat_entropy_z",
    "dataset_std_cat_entropy_z",
    "dataset_mean_top_freq_z",
    "dataset_max_top_freq_z",
}

LANDMARKING_DATASET_FEATURES = {
    "dataset_landmark_acc_knn1_z",
    "dataset_landmark_acc_gaussian_nb_z",
    "dataset_landmark_acc_decision_stump_z",
    "dataset_landmark_acc_logreg_z",
}

DEFAULT_EXCLUDED_FEATURE_COLUMNS = {
    "run_id",
    "dataset",
    "model",
    "client_id",
    "client_numeric_id",
    "split",
    "selection_id",
    "shard_id",
    "explainer_name",
    "explainer_type",
    "config_id",
    "instance_id",
    "dataset_index",
    "instance_index",
    "instance_index_within_job",
    "true_label",
    "prediction",
    "explained_class",
    "method",
    "method_variant",
    "source_instance_metrics_path",
    "is_pareto_optimal",
    "candidate_index_within_instance",
}


@dataclass(frozen=True)
class PairwiseRecommenderData:
    """Client-local data for training and evaluating a recommender."""

    feature_columns: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    pair_count: int
    augmented_pair_count: int
    candidate_count: int
    instance_count: int


@dataclass(frozen=True)
class RecommenderInstanceSplit:
    """Deterministic train/test split over recommender instance identifiers."""

    train_instance_ids: tuple[int, ...]
    test_instance_ids: tuple[int, ...]


def infer_recommender_feature_columns(
    candidates: pd.DataFrame,
    *,
    excluded_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Infer numeric candidate feature columns used by the recommender model."""

    exclusions = set(DEFAULT_EXCLUDED_FEATURE_COLUMNS)
    exclusions.update(STATISTICAL_DATASET_FEATURES)
    exclusions.update(LANDMARKING_DATASET_FEATURES)
    exclusions.update(excluded_columns or ())
    numeric_columns = candidates.select_dtypes(include=["number", "bool"]).columns
    feature_columns = tuple(
        str(column)
        for column in numeric_columns
        if str(column) not in exclusions
    )
    if not feature_columns:
        raise ValueError("No numeric recommender feature columns were found.")
    return feature_columns


def split_recommender_instance_ids(
    candidates: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RecommenderInstanceSplit:
    """Split unique dataset_index values into train/test instance ids."""

    if "dataset_index" not in candidates.columns:
        raise ValueError("Candidates are missing required column: 'dataset_index'.")
    instance_ids = sorted(
        int(value)
        for value in candidates["dataset_index"].dropna().unique().tolist()
    )
    if len(instance_ids) < 2:
        raise ValueError("At least two recommender instances are required to perform a split.")
    train_ids, test_ids = train_test_split(
        instance_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    resolved_train_ids = tuple(sorted(int(value) for value in train_ids))
    resolved_test_ids = tuple(sorted(int(value) for value in test_ids))
    if not resolved_train_ids or not resolved_test_ids:
        raise ValueError("Recommender instance split must produce non-empty train and test partitions.")
    return RecommenderInstanceSplit(
        train_instance_ids=resolved_train_ids,
        test_instance_ids=resolved_test_ids,
    )


def build_pairwise_recommender_data(
    *,
    candidates: pd.DataFrame,
    pair_labels: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    augment_symmetric: bool = True,
) -> PairwiseRecommenderData:
    """Convert candidate features and pair labels into logistic training arrays.

    The returned `X` rows are `features(pair_1) - features(pair_2)`.
    Labels are binary:
    - `1`: pair_1 preferred
    - `0`: pair_2 preferred

    When `augment_symmetric=True`, every row is mirrored as `-X` with the
    opposite label. This removes dependence on arbitrary pair ordering and keeps
    the pairwise class balance stable.
    """

    _validate_candidate_and_label_frames(candidates, pair_labels)
    resolved_features = tuple(
        feature_columns
        if feature_columns is not None
        else infer_recommender_feature_columns(candidates)
    )
    missing_features = [column for column in resolved_features if column not in candidates.columns]
    if missing_features:
        raise ValueError(f"Candidates are missing feature columns: {missing_features}")

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for dataset_index, pair_group in pair_labels.groupby("dataset_index", sort=True, dropna=False):
        candidate_group = candidates.loc[candidates["dataset_index"] == dataset_index]
        if candidate_group.empty:
            LOGGER.warning("Skipping labels for missing dataset_index=%s.", dataset_index)
            continue
        candidate_matrix = _candidate_feature_matrix(candidate_group, resolved_features)
        for _, row in pair_group.iterrows():
            pair_1 = str(row["pair_1"])
            pair_2 = str(row["pair_2"])
            raw_label = row["label"]
            if pair_1 == pair_2:
                LOGGER.warning("Skipping degenerate pair (%s, %s).", pair_1, pair_2)
                continue
            if pair_1 not in candidate_matrix.index or pair_2 not in candidate_matrix.index:
                LOGGER.warning(
                    "Skipping pair (%s, %s) because a variant is missing from candidates.",
                    pair_1,
                    pair_2,
                )
                continue
            if raw_label not in (0, 1):
                LOGGER.warning("Skipping pair with unexpected label=%s.", raw_label)
                continue
            diff = (
                candidate_matrix.loc[pair_1].to_numpy(dtype=float)
                - candidate_matrix.loc[pair_2].to_numpy(dtype=float)
            )
            label = 1 if int(raw_label) == 0 else 0
            rows.append(diff)
            labels.append(label)
            if augment_symmetric:
                rows.append(-diff)
                labels.append(1 - label)

    if not rows:
        raise ValueError("No valid pairwise recommender rows could be built.")
    X = np.vstack(rows).astype(np.float64, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    return PairwiseRecommenderData(
        feature_columns=resolved_features,
        X=X,
        y=y,
        pair_count=int(len(pair_labels)),
        augmented_pair_count=int(X.shape[0]),
        candidate_count=int(len(candidates)),
        instance_count=int(candidates["dataset_index"].nunique()),
    )


def _candidate_feature_matrix(
    candidates: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    deduped = _dedupe_candidates_by_variant(candidates)
    matrix = deduped.set_index("method_variant")[list(feature_columns)]
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if not matrix.index.is_unique:
        matrix = matrix.groupby(level=0).mean()
    return matrix


def _dedupe_candidates_by_variant(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or "method_variant" not in candidates.columns:
        return candidates
    method_variant = candidates["method_variant"].astype(str)
    if method_variant.is_unique:
        return candidates.assign(method_variant=method_variant)

    candidates = candidates.copy()
    candidates["method_variant"] = method_variant
    grouped = candidates.groupby("method_variant", sort=False)
    numeric_cols = candidates.select_dtypes(include=["number", "bool"]).columns.tolist()
    first_cols = [
        column
        for column in candidates.columns
        if column not in numeric_cols and column != "method_variant"
    ]

    parts: list[pd.DataFrame] = []
    if first_cols:
        parts.append(grouped[first_cols].first())
    if numeric_cols:
        parts.append(grouped[numeric_cols].mean())
    collapsed = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=grouped.size().index)
    collapsed.insert(0, "method_variant", collapsed.index.astype(str))
    return collapsed.reset_index(drop=True)


def _validate_candidate_and_label_frames(
    candidates: pd.DataFrame,
    pair_labels: pd.DataFrame,
) -> None:
    candidate_required = {"dataset_index", "method_variant"}
    label_required = {"dataset_index", "pair_1", "pair_2", "label"}
    missing_candidate = candidate_required - set(candidates.columns)
    missing_labels = label_required - set(pair_labels.columns)
    if missing_candidate:
        raise ValueError(f"Candidates are missing required columns: {sorted(missing_candidate)}")
    if missing_labels:
        raise ValueError(f"Pair labels are missing required columns: {sorted(missing_labels)}")
    if candidates.empty:
        raise ValueError("Candidate context is empty.")
    if pair_labels.empty:
        raise ValueError("Pair labels are empty.")
