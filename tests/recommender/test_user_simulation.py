from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from fed_perso_xai.recommender.user_simulation import (
    DirichletPersonaSimulator,
    PersonaConfig,
    label_recommender_context,
)
from fed_perso_xai.utils.config import ArtifactPaths

PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


def _persona() -> PersonaConfig:
    return PersonaConfig.from_dict(
        {
            "persona": "unit",
            "type": "flat_dirichlet",
            "tau": 0.05,
            "properties": {
                "quality": {
                    "preference": 1,
                    "metrics": ["quality", "missing_metric"],
                }
            },
        }
    )


def test_dirichlet_persona_labels_metric_z_columns_and_reports_missing_metrics() -> None:
    simulator = DirichletPersonaSimulator(
        _persona(),
        seed=0,
        label_seed=0,
        concentration_c=1.0,
        tau=0.01,
    )
    simulator.metric_weights = {"quality": 1.0, "missing_metric": 0.0}
    candidates = pd.DataFrame(
        {
            "client_id": ["client_000", "client_000"],
            "dataset_index": [7, 7],
            "instance_id": ["row-7", "row-7"],
            "method_variant": ["better", "worse"],
            "metric_quality_z": [5.0, -5.0],
        }
    )

    labels, metadata = simulator.label_client_candidates(candidates)

    assert len(labels) == 1
    assert labels.iloc[0]["pair_1"] == "better"
    assert labels.iloc[0]["pair_2"] == "worse"
    assert int(labels.iloc[0]["label"]) == 0
    assert labels.iloc[0]["probability_pair_1_preferred"] == pytest.approx(1.0)
    assert metadata["active_metrics"] == ["quality"]
    assert metadata["missing_configured_metrics"] == ["missing_metric"]


def test_dirichlet_persona_rejects_context_without_active_metrics() -> None:
    simulator = DirichletPersonaSimulator(_persona(), seed=0, concentration_c=1.0)
    candidates = pd.DataFrame(
        {
            "dataset_index": [0, 0],
            "method_variant": ["a", "b"],
            "some_other_feature": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="No persona metrics"):
        simulator.label_client_candidates(candidates)


def test_dirichlet_persona_dedupes_duplicate_variants() -> None:
    simulator = DirichletPersonaSimulator(_persona(), seed=0, label_seed=0, concentration_c=1.0)
    simulator.metric_weights = {"quality": 1.0, "missing_metric": 0.0}
    candidates = pd.DataFrame(
        {
            "dataset_index": [0, 0, 0],
            "method_variant": ["a", "a", "b"],
            "metric_quality_z": [1.0, 1.0, 0.0],
        }
    )

    labels, _ = simulator.label_client_candidates(candidates)

    assert len(labels) == 1
    assert set(np.ravel(labels[["pair_1", "pair_2"]].to_numpy())) == {"a", "b"}


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_label_recommender_context_uses_client_stable_seeds_for_subset_reruns(tmp_path) -> None:
    paths = ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )
    run_id = "unit-run"
    selection_id = "selection-0"
    label_filename = "custom_pairwise_labels.parquet"
    persona_path = tmp_path / "persona.yaml"
    persona_path.write_text(
        "\n".join(
            [
                "persona: unit",
                "type: flat_dirichlet",
                "tau: 1.0",
                "properties:",
                "  quality:",
                "    preference: 1.0",
                "    metrics: [quality]",
            ]
        ),
        encoding="utf-8",
    )

    run_dir = paths.federated_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    for client_id in ("client_000", "client_001"):
        context_dir = run_dir / "clients" / client_id / "recommender_context" / selection_id
        context_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "client_id": [client_id] * 3,
                "dataset_index": [0, 0, 0],
                "instance_id": ["row-0"] * 3,
                "method_variant": ["a", "b", "c"],
                "metric_quality_z": [0.2, 0.0, -0.2],
            }
        ).to_parquet(context_dir / "candidate_context.parquet", index=False)

    label_recommender_context(
        run_id=run_id,
        selection_id=selection_id,
        persona="unit",
        persona_config_path=persona_path,
        label_filename=label_filename,
        seed=42,
        label_seed=1729,
        concentration_c=1.0,
        paths=paths,
    )

    client_dir = run_dir / "clients" / "client_001" / "recommender_labels" / selection_id / "unit"
    first_labels_path = client_dir / label_filename
    first_labels = pd.read_parquet(first_labels_path)
    first_metadata = json.loads((client_dir / "simulation_metadata.json").read_text(encoding="utf-8"))
    assert not (client_dir / "pairwise_labels.parquet").exists()
    assert first_metadata["pairwise_labels"] == str(first_labels_path)

    label_recommender_context(
        run_id=run_id,
        selection_id=selection_id,
        persona="unit",
        persona_config_path=persona_path,
        clients="client_001",
        label_filename=label_filename,
        seed=42,
        label_seed=1729,
        concentration_c=1.0,
        paths=paths,
    )

    second_labels = pd.read_parquet(first_labels_path)
    second_metadata = json.loads((client_dir / "simulation_metadata.json").read_text(encoding="utf-8"))

    pd.testing.assert_frame_equal(first_labels, second_labels)
    assert first_metadata["seed"] == second_metadata["seed"]
    assert first_metadata["label_seed"] == second_metadata["label_seed"]
