"""Shared client-local background-data helpers for local explainers."""

from __future__ import annotations

from typing import Any

import numpy as np


def require_client_local_background(expl_cfg: dict[str, Any]) -> str:
    """Validate the currently supported background-data source."""

    background_source = str(
        expl_cfg.get("background_data_source", "client_local_train")
    ).strip()
    if background_source != "client_local_train":
        raise ValueError(
            "Only client-local explainer background data is implemented. "
            "Set experiment.explanation.background_data_source to 'client_local_train'."
        )
    return background_source


def sample_client_local_background(
    X: np.ndarray,
    *,
    expl_cfg: dict[str, Any],
    random_state: int | None,
) -> np.ndarray:
    """Sample a reproducible client-local background/reference dataset."""

    X_np = np.asarray(X, dtype=np.float64)
    if X_np.ndim != 2:
        raise ValueError("Background data must be a 2D array.")
    if X_np.shape[0] == 0:
        raise ValueError("Background data cannot be sampled from an empty dataset.")

    require_client_local_background(expl_cfg)
    sample_size = int(expl_cfg.get("background_sample_size", 100))
    sample_size = min(sample_size, X_np.shape[0])

    rng = np.random.default_rng(random_state)
    indices = rng.choice(X_np.shape[0], size=sample_size, replace=False)
    return X_np[indices]


def build_client_local_mean_reference(
    X: np.ndarray,
    *,
    expl_cfg: dict[str, Any],
) -> np.ndarray:
    """Build a reproducible client-local reference vector for IG-style methods."""

    X_np = np.asarray(X, dtype=np.float64)
    if X_np.ndim != 2:
        raise ValueError("Reference data must be a 2D array.")
    if X_np.shape[0] == 0:
        raise ValueError("Reference data cannot be built from an empty dataset.")

    require_client_local_background(expl_cfg)
    return np.mean(X_np, axis=0)
