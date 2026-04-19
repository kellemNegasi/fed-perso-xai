"""Run-manifest and reproducibility helpers."""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn


def build_run_id(
    *,
    experiment_type: str,
    dataset_name: str,
    seed: int,
    num_clients: int | None = None,
    alpha: float | None = None,
) -> str:
    """Build a stable run identifier."""

    parts = [experiment_type, dataset_name]
    if num_clients is not None:
        parts.append(f"{num_clients}clients")
    if alpha is not None:
        parts.append(f"alpha{float(alpha)}")
    parts.append(f"seed{seed}")
    return "-".join(parts)


def build_reproducibility_metadata(*, seed: int) -> dict[str, Any]:
    """Return environment metadata needed to reproduce a run."""

    package_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for package_name, output_key in (("flwr", "flwr"), ("ray", "ray")):
        try:
            package_versions[output_key] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[output_key] = None

    return {
        "seed": int(seed),
        "platform": platform.platform(),
        "package_versions": package_versions,
    }


def relative_artifact_path(path: Path, root_dir: Path) -> str:
    """Return a stable relative path for manifest entries."""

    return str(path.relative_to(root_dir))
