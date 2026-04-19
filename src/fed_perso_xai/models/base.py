"""Generic tabular classifier interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class TabularClassifier(Protocol):
    """Protocol shared by centralized and federated stage-1 models."""

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> float:
        """Fit on a training split and return the final training loss."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities."""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return binary cross-entropy loss."""

    def save(self, path: Path) -> Path:
        """Persist the model artifact."""

    def get_parameters(self) -> list[np.ndarray]:
        """Return parameters in Flower-compatible order."""

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load parameters in Flower-compatible order."""
