"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed the Python and NumPy RNGs and return a generator."""

    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
