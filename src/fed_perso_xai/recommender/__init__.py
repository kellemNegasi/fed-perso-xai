"""Recommender data preparation and user-simulation utilities."""

from fed_perso_xai.recommender.data import (
    PairwiseRecommenderData,
    build_pairwise_recommender_data,
    infer_recommender_feature_columns,
)
from fed_perso_xai.recommender.model import (
    PairwiseLogisticConfig,
    PairwiseLogisticRecommender,
    initialize_recommender_parameters,
    load_pairwise_logistic_recommender,
)
from fed_perso_xai.recommender.evaluation import (
    build_ground_truth_order,
    evaluate_ranked_scores,
)
from fed_perso_xai.recommender.user_simulation import (
    DEFAULT_USER_SIMULATOR_REGISTRY,
    DirichletPersonaSimulator,
    PairwiseLabelConfig,
    PersonaConfig,
    UserSimulator,
    label_recommender_context,
    load_persona_config,
)

__all__ = [
    "DEFAULT_USER_SIMULATOR_REGISTRY",
    "DirichletPersonaSimulator",
    "PairwiseLogisticConfig",
    "PairwiseLogisticRecommender",
    "PairwiseLabelConfig",
    "PairwiseRecommenderData",
    "PersonaConfig",
    "UserSimulator",
    "build_pairwise_recommender_data",
    "infer_recommender_feature_columns",
    "build_ground_truth_order",
    "evaluate_ranked_scores",
    "load_pairwise_logistic_recommender",
    "initialize_recommender_parameters",
    "label_recommender_context",
    "load_persona_config",
]
