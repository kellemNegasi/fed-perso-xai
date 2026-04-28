"""Evaluation helpers for pairwise explanation recommenders."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def is_recommender_metric_key(key: object) -> bool:
    """Return whether a key is an aggregate recommender quality metric."""

    if not isinstance(key, str):
        return False
    return key == "pearson" or key.startswith("precision_at_")


@dataclass(frozen=True)
class RecommenderEvaluationResult:
    """Aggregate and per-client recommender evaluation metrics."""

    aggregate: dict[str, float]
    clients: list[dict[str, object]]


def compute_pairwise_copeland_scores(pair_labels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pairwise labels into Copeland-style scores by method variant."""

    required = {"pair_1", "pair_2", "label"}
    missing = required - set(pair_labels.columns)
    if missing:
        raise ValueError(f"Pair labels are missing required columns: {sorted(missing)}")

    wins: Counter[str] = Counter()
    losses: Counter[str] = Counter()
    for _, row in pair_labels.iterrows():
        pair_1 = str(row.get("pair_1"))
        pair_2 = str(row.get("pair_2"))
        label = row.get("label")
        if pair_1 == pair_2 or label not in (0, 1):
            continue
        if int(label) == 0:
            winner, loser = pair_1, pair_2
        else:
            winner, loser = pair_2, pair_1
        wins[winner] += 1
        losses[loser] += 1

    variants = sorted(set(wins) | set(losses))
    rows = [
        {
            "method_variant": variant,
            "wins": int(wins.get(variant, 0)),
            "losses": int(losses.get(variant, 0)),
            "score": int(wins.get(variant, 0) - losses.get(variant, 0)),
        }
        for variant in variants
    ]
    return pd.DataFrame(rows)


def build_ground_truth_order(pair_labels: pd.DataFrame) -> list[str]:
    """Convert pairwise labels into one deterministic global order."""

    scores = compute_pairwise_copeland_scores(pair_labels)
    if scores.empty:
        return []
    ranked = scores.sort_values(
        by=["score", "wins", "method_variant"],
        ascending=[False, False, True],
    )
    return ranked["method_variant"].astype(str).tolist()


def precision_at_k(
    predicted_order: Sequence[str],
    ground_truth_order: Sequence[str],
    k: int,
) -> float:
    """Return top-k overlap precision with a denominator capped by available items."""

    if not predicted_order or not ground_truth_order:
        return 0.0
    limit = min(max(1, int(k)), len(predicted_order), len(ground_truth_order))
    pred_top = set(str(item) for item in predicted_order[:limit])
    truth_top = set(str(item) for item in ground_truth_order[:limit])
    return float(len(pred_top & truth_top) / limit)


def pearson_rank_correlation(
    predicted_scores: Mapping[str, float],
    ground_truth_order: Sequence[str],
) -> float:
    """Compute Pearson correlation between predicted and ground-truth rank positions."""

    predicted_order = order_scores(predicted_scores)
    pred_rank = {variant: idx for idx, variant in enumerate(predicted_order)}
    truth_rank = {str(variant): idx for idx, variant in enumerate(ground_truth_order)}
    variants = sorted(set(pred_rank) & set(truth_rank))
    if len(variants) < 2:
        return 0.0
    pred = np.asarray([pred_rank[variant] for variant in variants], dtype=float)
    truth = np.asarray([truth_rank[variant] for variant in variants], dtype=float)
    if float(np.std(pred)) == 0.0 or float(np.std(truth)) == 0.0:
        return 0.0
    corr = float(np.corrcoef(pred, truth)[0, 1])
    return 0.0 if not np.isfinite(corr) else corr


def order_scores(predicted_scores: Mapping[str, float]) -> list[str]:
    """Order predicted scores descending with deterministic variant tie-breaking."""

    def sort_key(item: tuple[str, float]) -> tuple[float, str]:
        variant, score = item
        value = float(score) if np.isfinite(float(score)) else float("-inf")
        return (-value, str(variant))

    return [variant for variant, _ in sorted(predicted_scores.items(), key=sort_key)]


def evaluate_ranked_scores(
    *,
    predicted_scores: Mapping[str, float],
    pair_labels: pd.DataFrame,
    top_k: Iterable[int] = (1, 3, 5),
) -> dict[str, object]:
    """Evaluate predicted variant scores against pair-label-derived global order."""

    ground_truth = build_ground_truth_order(pair_labels)
    predicted_order = order_scores(predicted_scores)
    metrics: dict[str, object] = {
        "ground_truth_order": ground_truth,
        "predicted_order": predicted_order,
        "pearson": pearson_rank_correlation(predicted_scores, ground_truth),
        "variant_count": int(len(set(ground_truth) & set(predicted_order))),
    }
    for k in top_k:
        metrics[f"precision_at_{int(k)}"] = precision_at_k(predicted_order, ground_truth, int(k))
    return metrics


def evaluate_grouped_ranked_scores(
    *,
    candidate_scores: pd.DataFrame,
    pair_labels: pd.DataFrame,
    top_k: Iterable[int] = (1, 3, 5),
) -> dict[str, object]:
    """Evaluate ranked candidate scores independently within each dataset_index."""

    required_candidate_columns = {"dataset_index", "method_variant", "score"}
    missing_candidate_columns = required_candidate_columns - set(candidate_scores.columns)
    if missing_candidate_columns:
        raise ValueError(
            f"Candidate scores are missing required columns: {sorted(missing_candidate_columns)}"
        )
    if "dataset_index" not in pair_labels.columns:
        raise ValueError("Pair labels are missing required column: ['dataset_index']")

    instance_metrics: list[dict[str, object]] = []
    for dataset_index, pair_group in pair_labels.groupby("dataset_index", sort=True, dropna=False):
        candidate_group = candidate_scores.loc[candidate_scores["dataset_index"] == dataset_index]
        if candidate_group.empty:
            LOGGER.warning("Skipping evaluation for missing dataset_index=%s.", dataset_index)
            continue
        predicted_scores = (
            candidate_group.groupby("method_variant", sort=True)["score"].mean().to_dict()
        )
        metrics = evaluate_ranked_scores(
            predicted_scores={str(key): float(value) for key, value in predicted_scores.items()},
            pair_labels=pair_group,
            top_k=top_k,
        )
        instance_row: dict[str, object] = {
            "dataset_index": int(dataset_index),
            "candidate_count": int(candidate_group["method_variant"].astype(str).nunique()),
            "pair_count": int(len(pair_group)),
            "variant_count": int(metrics.get("variant_count", 0)),
            "pearson": float(metrics.get("pearson", 0.0)),
            "ground_truth_order": list(metrics.get("ground_truth_order", [])),
            "predicted_order": list(metrics.get("predicted_order", [])),
        }
        for key, value in metrics.items():
            if key.startswith("precision_at_"):
                instance_row[key] = float(value)
        instance_metrics.append(instance_row)

    return {
        "instance_count": int(len(instance_metrics)),
        "pair_count": int(sum(int(item["pair_count"]) for item in instance_metrics)),
        "aggregate": aggregate_client_metrics(instance_metrics),
        "instances": instance_metrics,
    }


def aggregate_client_metrics(clients: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Average numeric client metrics, weighted by each client's labeled pair count."""

    weighted_sums: dict[str, float] = {}
    weights: dict[str, float] = {}
    for row in clients:
        weight = float(row.get("pair_count", 0) or 0)
        if weight <= 0:
            weight = 1.0
        for key, value in row.items():
            if not is_recommender_metric_key(key) or not isinstance(value, (int, float)):
                continue
            weighted_sums[key] = weighted_sums.get(key, 0.0) + float(value) * weight
            weights[key] = weights.get(key, 0.0) + weight
    return {
        key: weighted_sums[key] / weights[key]
        for key in sorted(weighted_sums)
        if weights.get(key, 0.0) > 0.0
    }
