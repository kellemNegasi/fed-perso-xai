"""Run-artifact helpers keyed by stable federated run_id values."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fed_perso_xai.utils.config import ArtifactPaths
from fed_perso_xai.utils.paths import federated_run_artifact_dir, federated_run_metadata_path


@dataclass(frozen=True)
class FederatedRunContext:
    """Resolved run-level metadata and canonical paths for one run_id."""

    run_id: str
    run_artifact_dir: Path
    run_metadata_path: Path
    metadata: dict[str, Any]

    @property
    def model_artifact_path(self) -> Path:
        return self.run_artifact_dir / str(self.metadata["model_artifact_path"])

    @property
    def model_metadata_path(self) -> Path:
        return self.run_artifact_dir / str(self.metadata["model_metadata_path"])

    @property
    def training_metadata_path(self) -> Path:
        return self.run_artifact_dir / str(self.metadata["training_metadata_path"])

    @property
    def partition_root(self) -> Path:
        return Path(str(self.metadata["partition_reference"]["partition_data_root"]))

    @property
    def feature_metadata_path(self) -> Path:
        feature_ref = self.metadata.get("feature_metadata_path")
        if feature_ref:
            return self.run_artifact_dir / str(feature_ref)
        return Path(str(self.metadata["partition_reference"]["feature_metadata_path"]))


def resolve_federated_run_context(
    *,
    paths: ArtifactPaths,
    run_id: str,
) -> FederatedRunContext:
    """Resolve the run-addressable artifact directory and run metadata."""

    run_artifact_dir = federated_run_artifact_dir(paths, run_id)
    run_metadata_path = federated_run_metadata_path(run_artifact_dir)
    if not run_metadata_path.exists():
        raise FileNotFoundError(
            f"Run '{run_id}' is not registered at '{run_metadata_path}'."
        )
    metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    return FederatedRunContext(
        run_id=run_id,
        run_artifact_dir=run_artifact_dir,
        run_metadata_path=run_metadata_path,
        metadata=metadata,
    )
