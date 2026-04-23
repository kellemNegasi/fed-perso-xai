"""Run-manifest and reproducibility helpers."""

from __future__ import annotations

from datetime import UTC, datetime
import importlib.metadata
import hashlib
import json
import platform
import subprocess
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
    model_name: str | None = None,
    timestamp: str | None = None,
    run_defining_payload: dict[str, Any] | None = None,
) -> str:
    """Build a stable run identifier."""

    parts = [experiment_type, dataset_name]
    if timestamp:
        compact = timestamp.replace("-", "").replace(":", "").replace("+00:00", "z")
        compact = compact.replace("T", "t").replace(".", "")
        parts.append(compact.lower())
    if model_name:
        parts.append(model_name)
    if num_clients is not None:
        parts.append(f"{num_clients}clients")
    if alpha is not None:
        parts.append(f"alpha{float(alpha)}")
    parts.append(f"seed{seed}")
    if run_defining_payload:
        serialized = json.dumps(run_defining_payload, sort_keys=True, separators=(",", ":"))
        parts.append(hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12])
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


def current_utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(UTC).isoformat()


def resolve_git_commit_hash(start_path: Path) -> str | None:
    """Return the current Git commit hash for the nearest repository, if available."""

    repository_root = _find_repository_root(start_path.resolve())
    if repository_root is None:
        repository_root = _find_repository_root(Path.cwd())
    if repository_root is None:
        return None

    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    commit_hash = result.stdout.strip()
    return commit_hash or None


def relative_artifact_path(path: Path, root_dir: Path) -> str:
    """Return a stable relative path for manifest entries."""

    return str(path.relative_to(root_dir))


def _find_repository_root(start_path: Path) -> Path | None:
    current = start_path if start_path.is_dir() else start_path.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None
