"""Shared explanation-metric framework and future concrete evaluator package."""

from .attribution_utils import (
    FEATURE_ATTRIBUTION_FIELD_CANDIDATES,
    FEATURE_METHOD_KEYS,
    coerce_attribution_vector,
    extract_attribution_vector,
    extract_instance_vector,
    prepare_attributions,
)
from .base_metric import MetricCapabilities, MetricInput, MetricScoreMap
from .baselines import baseline_vector, dataset_feature_std, feature_scale
from .compactness import CompactnessEvaluator
from .completeness import CompletenessEvaluator
from .confidence import ConfidenceEvaluator
from .contrastivity import ContrastivityEvaluator
from .continuity import ContinuityEvaluator
from .correctness import CorrectnessEvaluator
from .covariate_complexity import CovariateComplexityEvaluator
from .execution import MetricExecutionResult, evaluate_metric, evaluate_metrics_for_method
from .factory import make_metric, metric_capabilities
from .infidelity import InfidelityEvaluator
from .monotonicity import MonotonicityEvaluator
from .non_sensitivity import NonSensitivityEvaluator
from .perturbation import (
    add_scaled_gaussian_noise,
    approximate_perturbed_attributions,
    build_metric_rng,
    chunk_indices,
    generate_random_masked_batch,
    mask_feature_indices,
    match_std_vector,
    sample_random_mask_indices,
    support_indices,
    top_k_mask_indices,
)
from .prediction_utils import (
    extract_prediction_value,
    model_prediction,
    model_predictions,
    prediction_label,
    prediction_value_from_probabilities,
)
from .registry import DEFAULT_METRIC_REGISTRY, MetricRegistry, load_metric_config
from .utils import (
    coerce_metric_dict,
    extract_metric_parameters,
    safe_scalar,
    structural_similarity,
    value_at,
)

__all__ = [
    "DEFAULT_METRIC_REGISTRY",
    "FEATURE_ATTRIBUTION_FIELD_CANDIDATES",
    "FEATURE_METHOD_KEYS",
    "CompactnessEvaluator",
    "CompletenessEvaluator",
    "ConfidenceEvaluator",
    "ContrastivityEvaluator",
    "ContinuityEvaluator",
    "CorrectnessEvaluator",
    "CovariateComplexityEvaluator",
    "InfidelityEvaluator",
    "MonotonicityEvaluator",
    "NonSensitivityEvaluator",
    "MetricCapabilities",
    "MetricExecutionResult",
    "MetricInput",
    "MetricRegistry",
    "MetricScoreMap",
    "add_scaled_gaussian_noise",
    "approximate_perturbed_attributions",
    "baseline_vector",
    "build_metric_rng",
    "chunk_indices",
    "coerce_attribution_vector",
    "coerce_metric_dict",
    "dataset_feature_std",
    "evaluate_metric",
    "evaluate_metrics_for_method",
    "extract_attribution_vector",
    "extract_instance_vector",
    "extract_metric_parameters",
    "extract_prediction_value",
    "feature_scale",
    "generate_random_masked_batch",
    "load_metric_config",
    "make_metric",
    "mask_feature_indices",
    "match_std_vector",
    "metric_capabilities",
    "model_prediction",
    "model_predictions",
    "prediction_label",
    "prediction_value_from_probabilities",
    "prepare_attributions",
    "safe_scalar",
    "sample_random_mask_indices",
    "structural_similarity",
    "support_indices",
    "top_k_mask_indices",
    "value_at",
]
