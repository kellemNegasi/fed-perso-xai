from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fed_perso_xai.recommender import (
    PairwiseLogisticConfig,
    PairwiseLogisticRecommender,
    build_pairwise_recommender_data,
    infer_recommender_feature_columns,
)


def test_build_pairwise_recommender_data_uses_pair_label_encoding() -> None:
    candidates = pd.DataFrame(
        {
            "dataset_index": [0, 0],
            "method_variant": ["a", "b"],
            "metric_quality_z": [2.0, -1.0],
            "candidate_index_within_instance": [0, 1],
        }
    )
    labels = pd.DataFrame(
        {
            "dataset_index": [0],
            "pair_1": ["a"],
            "pair_2": ["b"],
            "label": [0],
        }
    )

    data = build_pairwise_recommender_data(candidates=candidates, pair_labels=labels)

    assert data.feature_columns == ("metric_quality_z",)
    assert data.X.tolist() == [[3.0], [-3.0]]
    assert data.y.tolist() == [1, 0]
    assert data.augmented_pair_count == 2


def test_infer_recommender_feature_columns_excludes_identifiers() -> None:
    candidates = pd.DataFrame(
        {
            "dataset_index": [0],
            "prediction": [1],
            "is_pareto_optimal": [True],
            "candidate_index_within_instance": [0],
            "metric_quality_z": [0.5],
            "hp_alpha": [1.2],
        }
    )

    assert infer_recommender_feature_columns(candidates) == (
        "metric_quality_z",
        "hp_alpha",
    )


def test_pairwise_logistic_recommender_learns_separable_preferences() -> None:
    rng = np.random.default_rng(0)
    positive = rng.normal(loc=2.0, scale=0.2, size=(32, 1))
    negative = rng.normal(loc=-2.0, scale=0.2, size=(32, 1))
    X = np.vstack([positive, negative])
    y = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int64)

    model = PairwiseLogisticRecommender.from_config(
        n_features=1,
        config=PairwiseLogisticConfig(
            epochs=40,
            batch_size=8,
            learning_rate=0.2,
            l2_regularization=0.0,
        ),
    )
    initial_loss = model.loss(X, y)
    final_loss = model.fit(X, y, seed=7)

    assert final_loss < initial_loss
    assert model.predict_pairwise(np.asarray([[2.0], [-2.0]])).tolist() == [1, 0]


def test_pairwise_logistic_recommender_scores_candidates_without_bias() -> None:
    model = PairwiseLogisticRecommender.from_config(
        n_features=2,
        config=PairwiseLogisticConfig(),
    )
    model.set_parameters([np.asarray([2.0, -1.0]), np.asarray([5.0])])
    candidates = pd.DataFrame(
        {
            "method_variant": ["a", "b"],
            "x1": [1.0, 0.0],
            "x2": [0.0, 3.0],
        }
    )

    scores = model.score_candidates(candidates, ["x1", "x2"])

    assert scores.loc["a"] == pytest.approx(2.0)
    assert scores.loc["b"] == pytest.approx(-3.0)
