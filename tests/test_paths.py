from __future__ import annotations

from pathlib import Path

import pytest

from fed_perso_xai.utils.config import ArtifactPaths
from fed_perso_xai.utils.paths import federated_run_artifact_dir


def _build_paths(tmp_path: Path) -> ArtifactPaths:
    return ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )


@pytest.mark.parametrize(
    ("run_id", "message"),
    [
        ("", "must not be empty"),
        (".", "must not be '.' or '..'"),
        ("..", "must not be '.' or '..'"),
        ("../outside", "without path separators"),
        ("nested/run", "without path separators"),
        (r"nested\\run", "without path separators"),
    ],
)
def test_federated_run_artifact_dir_rejects_unsafe_run_ids(
    tmp_path: Path,
    run_id: str,
    message: str,
) -> None:
    paths = _build_paths(tmp_path)

    with pytest.raises(ValueError, match=message):
        federated_run_artifact_dir(paths, run_id)


def test_federated_run_artifact_dir_accepts_single_safe_segment(tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)

    run_dir = federated_run_artifact_dir(
        paths,
        "federated-training-adult_income-20260424t120000z-logistic_regression-seed7",
    )

    assert run_dir == paths.federated_root / "runs" / (
        "federated-training-adult_income-20260424t120000z-logistic_regression-seed7"
    )
