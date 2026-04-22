"""Stage-oriented orchestration entrypoints."""
from .stage_b_training import train_federated_stage_b
from .explanations import (
    ExplainerModelAdapter,
    LocalExplanationDataset,
    generate_client_local_explanations,
    instantiate_explainer,
    load_client_data_for_explanations,
    load_feature_names_from_metadata,
    load_saved_model_for_explanations,
    resolve_feature_names_for_explanations,
    save_client_explanations,
)

__all__ = [
    "ExplainerModelAdapter",
    "LocalExplanationDataset",
    "generate_client_local_explanations",
    "instantiate_explainer",
    "load_client_data_for_explanations",
    "load_feature_names_from_metadata",
    "load_saved_model_for_explanations",
    "resolve_feature_names_for_explanations",
    "save_client_explanations",
    "train_federated_stage_b",
]
