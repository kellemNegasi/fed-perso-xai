from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fed_perso_xai.recommender import (
    DEFAULT_RECOMMENDER_TYPE,
    PairwiseLogisticConfig,
    PairwiseLogisticRecommender,
    SVMRankRecommender,
    create_recommender,
    evaluate_grouped_ranked_scores,
    load_recommender,
)


def test_create_recommender_defaults_to_svm_rank() -> None:
    model = create_recommender(
        recommender_type=DEFAULT_RECOMMENDER_TYPE,
        n_features=1,
        config=PairwiseLogisticConfig(),
    )

    assert isinstance(model, SVMRankRecommender)


@pytest.mark.parametrize(
    ("recommender_type", "expected_type"),
    [
        ("svm_rank", SVMRankRecommender),
        ("pairwise_logistic", PairwiseLogisticRecommender),
    ],
)
def test_recommender_save_load_round_trip(tmp_path, recommender_type: str, expected_type: type) -> None:
    rng = np.random.default_rng(0)
    positive = rng.normal(loc=2.0, scale=0.2, size=(32, 1))
    negative = rng.normal(loc=-2.0, scale=0.2, size=(32, 1))
    X = np.vstack([positive, negative])
    y = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int64)

    model = create_recommender(
        recommender_type=recommender_type,
        n_features=1,
        config=PairwiseLogisticConfig(
            epochs=40,
            batch_size=8,
            learning_rate=0.2,
            l2_regularization=0.0,
        ),
    )
    model.fit(X, y, seed=7)

    path = tmp_path / f"{recommender_type}.npz"
    model.save(path)
    loaded = load_recommender(path)

    assert isinstance(loaded, expected_type)
    np.testing.assert_allclose(loaded.get_parameters()[0], model.get_parameters()[0])
    np.testing.assert_allclose(loaded.get_parameters()[1], model.get_parameters()[1])
    assert loaded.predict_pairwise(np.asarray([[2.0], [-2.0]])).tolist() == [1, 0]


@pytest.mark.parametrize("recommender_type", ["svm_rank", "pairwise_logistic"])
def test_recommender_scores_work_with_grouped_evaluation_pipeline(recommender_type: str) -> None:
    model = create_recommender(
        recommender_type=recommender_type,
        n_features=1,
        config=PairwiseLogisticConfig(
            epochs=40,
            batch_size=4,
            learning_rate=0.2,
            l2_regularization=0.0,
        ),
    )
    train_X = np.asarray([[2.0], [1.5], [-1.5], [-2.0]], dtype=np.float64)
    train_y = np.asarray([1, 1, 0, 0], dtype=np.int64)
    model.fit(train_X, train_y, seed=3)

    candidates = pd.DataFrame(
        {
            "dataset_index": [0, 0, 1, 1],
            "method_variant": ["a", "b", "a", "b"],
            "metric_quality_z": [2.0, -1.0, 1.5, -1.5],
        }
    )
    score_frame = candidates.loc[:, ["dataset_index", "method_variant"]].copy()
    score_frame["score"] = model.score_candidates(candidates, ["metric_quality_z"]).to_numpy()
    labels = pd.DataFrame(
        {
            "dataset_index": [0, 1],
            "pair_1": ["a", "a"],
            "pair_2": ["b", "b"],
            "label": [0, 0],
        }
    )

    metrics = evaluate_grouped_ranked_scores(
        candidate_scores=score_frame,
        pair_labels=labels,
        top_k=(1, 2, 8),
    )

    assert metrics["aggregate"]["precision_at_1"] == pytest.approx(1.0)
    assert metrics["aggregate"]["precision_at_8"] == pytest.approx(1.0)
    assert metrics["aggregate"]["pearson"] == pytest.approx(1.0)
    assert metrics["aggregate"]["pearson_at_1"] == pytest.approx(1.0)
    assert metrics["aggregate"]["pearson_at_8"] == pytest.approx(1.0)
    assert "dataset_index" not in metrics["aggregate"]
