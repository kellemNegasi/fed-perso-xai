"""
Shared masking / perturbation helpers used by explanation metrics.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def build_metric_rng(
    random_state: int | None,
    *,
    offset: int = 0,
) -> np.random.Generator:
    """Return a deterministic RNG, optionally offset per explanation index."""
    seed = None if random_state is None else int(random_state) + int(offset)
    return np.random.default_rng(seed)


def mask_feature_indices(
    instance: np.ndarray,
    indices: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """Replace selected feature indices with their baseline values."""
    perturbed = np.asarray(instance, dtype=float).reshape(-1).copy()
    baseline_vec = np.asarray(baseline, dtype=float).reshape(-1)
    feature_indices = np.asarray(indices, dtype=int).reshape(-1)
    perturbed[feature_indices] = baseline_vec[feature_indices]
    return perturbed


def top_k_mask_indices(importance: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the k highest-magnitude attribution scores."""
    vec = np.asarray(importance, dtype=float).reshape(-1)
    if vec.size == 0:
        return np.asarray([], dtype=int)
    k = max(0, min(int(k), vec.size))
    if k == 0:
        return np.asarray([], dtype=int)
    return np.argsort(-np.abs(vec))[:k]


def support_indices(
    importance: np.ndarray,
    *,
    magnitude_threshold: float,
    min_features: int,
) -> np.ndarray:
    """Return attribution-support indices based on thresholding and min-cardinality."""
    magnitudes = np.abs(np.asarray(importance, dtype=float).reshape(-1))
    mask = magnitudes >= float(magnitude_threshold)
    indices = np.flatnonzero(mask)
    if indices.size >= int(min_features):
        return indices

    order = np.argsort(-magnitudes)
    needed = max(int(min_features), 1)
    return order[: min(needed, magnitudes.size)]


def sample_random_mask_indices(
    rng: np.random.Generator,
    *,
    n_features: int,
    mask_size: int,
) -> np.ndarray:
    """Sample one random feature subset without replacement."""
    if mask_size <= 0 or n_features <= 0:
        return np.asarray([], dtype=int)
    if mask_size >= n_features:
        return np.arange(n_features, dtype=int)
    return np.asarray(rng.choice(n_features, size=mask_size, replace=False), dtype=int)


def generate_random_masked_batch(
    instance: np.ndarray,
    baseline: np.ndarray,
    *,
    n_trials: int,
    mask_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a batch of randomly masked copies of one instance."""
    inst = np.asarray(instance, dtype=float).reshape(-1)
    base = np.asarray(baseline, dtype=float).reshape(-1)
    batch = np.repeat(inst[np.newaxis, :], int(n_trials), axis=0)
    for row in range(int(n_trials)):
        indices = sample_random_mask_indices(
            rng,
            n_features=inst.size,
            mask_size=mask_size,
        )
        batch[row, indices] = base[indices]
    return batch


def chunk_indices(indices: np.ndarray, *, features_per_step: int) -> List[np.ndarray]:
    """Split indices into fixed-width groups for grouped perturbations."""
    groups: List[np.ndarray] = []
    step = max(1, int(features_per_step))
    for start in range(0, len(indices), step):
        groups.append(np.asarray(indices[start : start + step], dtype=int))
    return groups


def match_std_vector(feature_std: Optional[np.ndarray], n_features: int) -> np.ndarray:
    """Resize or broadcast the std vector so it matches the instance dimension."""
    if feature_std is None:
        return np.ones(n_features, dtype=float)
    std_vec = np.asarray(feature_std, dtype=float).reshape(-1)
    if std_vec.size == 1:
        return np.full(n_features, std_vec[0], dtype=float)
    if std_vec.size != n_features:
        return np.resize(std_vec, n_features)
    return std_vec


def add_scaled_gaussian_noise(
    instance: np.ndarray,
    *,
    feature_std: Optional[np.ndarray],
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add per-feature Gaussian noise scaled by dataset std statistics."""
    inst = np.asarray(instance, dtype=float).reshape(-1)
    std_vec = match_std_vector(feature_std, inst.shape[0])
    noise = rng.normal(0.0, std_vec * float(noise_scale), size=inst.shape[0])
    return inst + noise


def approximate_perturbed_attributions(
    original_instance: np.ndarray,
    perturbed_instance: np.ndarray,
    original_importance: np.ndarray,
) -> Optional[np.ndarray]:
    """Scale the original importance vector according to relative input changes."""
    if original_instance.shape != perturbed_instance.shape:
        return None

    # TODO: Replace this heuristic with a true re-run of the explainer on the perturbed
    # instance so continuity reflects the actual attribution change instead of the
    # assumed proportional scaling below:
    #   Δx_i = x'_i - x_i
    #   change_magnitude_i = |Δx_i| / (|x_i| + 1e-8)
    #   perturbed_importance_i = original_importance_i * (1 + 0.1 * change_magnitude_i)
    # The 0.1 factor was inherited from the prototype code — revisit whether that
    # damping constant still makes sense once we compute real perturbation scores.
    input_change = perturbed_instance - original_instance
    change_magnitude = np.abs(input_change) / (np.abs(original_instance) + 1e-8)
    return np.asarray(original_importance, dtype=float) * (1.0 + 0.1 * change_magnitude)
