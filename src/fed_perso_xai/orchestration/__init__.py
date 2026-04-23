"""High-level orchestration entrypoints."""
from .federated_training import train_federated_from_partitions
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
    "train_federated_from_partitions",
]
