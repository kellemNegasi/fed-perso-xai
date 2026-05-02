from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fed_perso_xai.recommender.user_simulation import (
    DirichletPersonaSimulator,
    PersonaConfig,
    assign_dirichlet_personas_to_clients,
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


def test_assign_dirichlet_personas_to_clients_is_deterministic_and_writes_artifact(tmp_path) -> None:
    client_ids = ["client_001", "client_000"]
    users_by_client = {
        "client_000": ["0", "1", "2", "3"],
        "client_001": ["0", "1", "2"],
    }
    output_path = tmp_path / "persona_assignment.json"

    first = assign_dirichlet_personas_to_clients(
        client_ids,
        users_by_client,
        alpha=0.3,
        seed=42,
        output_path=output_path,
    )
    second = assign_dirichlet_personas_to_clients(
        client_ids,
        users_by_client,
        alpha=0.3,
        seed=42,
    )

    assert first == second
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == first["artifact"]
    assert written["assignment_policy"] == "dirichlet_sampled"
    assert written["assignment_scope"] == "client"
    assert written["personas"] == ["lay", "clinician", "regulator"]
    assert sum(written["persona_probs"].values()) == pytest.approx(1.0)
    for client_id, payload in written["clients"].items():
        assert set(payload["persona_probs"]) == {"lay", "clinician", "regulator"}
        assert sum(payload["persona_probs"].values()) == pytest.approx(1.0)
        assert payload["sampled_client_persona"] in {"lay", "clinician", "regulator"}
        assert payload["argmax_persona"] in {"lay", "clinician", "regulator"}
        assert sum(payload["user_persona_counts"].values()) == len(users_by_client[client_id])
        assert payload["user_persona_counts"][payload["sampled_client_persona"]] == len(users_by_client[client_id])


def test_assign_dirichlet_personas_to_clients_changes_with_seed() -> None:
    client_ids = ["client_000", "client_001", "client_002"]
    users_by_client = {
        client_id: [str(idx) for idx in range(8)]
        for client_id in client_ids
    }

    first = assign_dirichlet_personas_to_clients(
        client_ids,
        users_by_client,
        alpha=0.3,
        seed=42,
    )
    second = assign_dirichlet_personas_to_clients(
        client_ids,
        users_by_client,
        alpha=0.3,
        seed=43,
    )

    assert first["artifact"] != second["artifact"]
    assert first["user_persona_by_client"] != second["user_persona_by_client"]


def test_label_recommender_context_rejects_unsafe_context_filename(tmp_path) -> None:
    paths = ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )

    with pytest.raises(ValueError, match=r"context_filename must be a single non-empty path segment\."):
        label_recommender_context(
            run_id="unit-run",
            selection_id="selection-0",
            persona="lay",
            context_filename="../candidate_context.parquet",
            paths=paths,
        )


def test_label_recommender_context_rejects_unsafe_persona_from_custom_config(tmp_path) -> None:
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
    persona = ".."
    persona_path = tmp_path / "persona.yaml"
    persona_path.write_text(
        "\n".join(
            [
                "persona: ..",
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

    with pytest.raises(ValueError, match=r"persona must be a single non-empty path segment\."):
        label_recommender_context(
            run_id=run_id,
            selection_id=selection_id,
            persona=persona,
            persona_config_path=persona_path,
            paths=paths,
        )


def test_label_recommender_context_rejects_custom_persona_config_mismatch(tmp_path) -> None:
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
    persona_path = tmp_path / "persona.yaml"
    persona_path.write_text(
        "\n".join(
            [
                "persona: expert",
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

    with pytest.raises(ValueError, match=r"Custom persona config mismatch: requested 'lay', got 'expert'\."):
        label_recommender_context(
            run_id=run_id,
            selection_id=selection_id,
            persona="lay",
            persona_config_path=persona_path,
            paths=paths,
        )


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
                "client_id": [client_id] * 6,
                "dataset_index": [0, 0, 0, 1, 1, 1],
                "instance_id": ["row-0"] * 3 + ["row-1"] * 3,
                "method_variant": ["a", "b", "c", "a", "b", "c"],
                "metric_quality_z": [0.2, 0.0, -0.2, 0.5, 0.1, -0.4],
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

    assert set(first_labels["split"]) == {"train", "test"}
    pd.testing.assert_frame_equal(first_labels, second_labels)
    assert first_metadata["seed"] == second_metadata["seed"]
    assert first_metadata["label_seed"] == second_metadata["label_seed"]
    assert first_metadata["instance_split"] == second_metadata["instance_split"]


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_label_recommender_context_supports_dirichlet_persona_assignment(tmp_path) -> None:
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
    run_dir = paths.federated_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    for client_id in ("client_000", "client_001"):
        context_dir = run_dir / "clients" / client_id / "recommender_context" / selection_id
        context_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "client_id": [client_id] * 9,
                "dataset_index": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                "instance_id": ["row-0"] * 3 + ["row-1"] * 3 + ["row-2"] * 3,
                "method_variant": ["a", "b", "c"] * 3,
                "metric_correctness_z": [0.3, 0.1, -0.2, 0.4, 0.0, -0.4, 0.2, -0.1, -0.3],
                "metric_infidelity_z": [0.0, -0.1, 0.3, 0.2, -0.2, 0.1, 0.1, -0.3, 0.4],
                "metric_consistency_z": [0.1, 0.2, -0.2, 0.3, -0.1, 0.0, 0.2, -0.2, 0.1],
                "metric_relative_input_stability_z": [0.2, 0.0, -0.1, 0.3, 0.1, -0.2, 0.4, 0.2, -0.3],
                "metric_covariate_complexity_z": [0.1, -0.1, 0.3, 0.2, -0.2, 0.4, 0.0, -0.3, 0.2],
                "metric_compactness_sparsity_z": [-0.1, 0.1, 0.4, -0.2, 0.0, 0.3, -0.3, 0.1, 0.2],
                "metric_contrastivity_z": [0.3, -0.1, 0.0, 0.1, 0.2, -0.3, 0.4, -0.2, 0.1],
                "metric_confidence_z": [0.2, 0.1, -0.2, 0.4, -0.1, 0.0, 0.3, -0.3, 0.2],
            }
        ).to_parquet(context_dir / "candidate_context.parquet", index=False)

    summary = label_recommender_context(
        run_id=run_id,
        selection_id=selection_id,
        persona="lay",
        output_persona="mixed",
        persona_assignment_policy="dirichlet_sampled",
        persona_assignment_alpha=0.3,
        seed=42,
        label_seed=1729,
        concentration_c=5.0,
        paths=paths,
    )

    assert summary["status"] == "labeled"
    assert summary["persona"] == "mixed"
    assert summary["persona_assignment_policy"] == "dirichlet_sampled"
    assignment_path = summary["persona_assignment"]["artifact_path"]
    assignment = json.loads(Path(assignment_path).read_text(encoding="utf-8"))
    assert assignment["personas"] == ["lay", "clinician", "regulator"]
    assert assignment["alpha"] == pytest.approx(0.3)

    labels_path = (
        run_dir
        / "clients"
        / "client_000"
        / "recommender_labels"
        / selection_id
        / "mixed"
        / "pairwise_labels.parquet"
    )
    metadata_path = labels_path.with_name("simulation_metadata.json")
    labels = pd.read_parquet(labels_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert "assigned_persona" in labels.columns
    assert set(labels["assigned_persona"]).issubset({"lay", "clinician", "regulator"})
    assert metadata["persona"] == "mixed"
    assert labels["assigned_persona"].nunique() == 1
    assert labels["assigned_persona"].iloc[0] == metadata["persona_assignment"]["sampled_client_persona"]
    assert metadata["persona_assignment"]["sampled_client_persona"] in {"lay", "clinician", "regulator"}
    assert sum(metadata["persona_assignment"]["user_persona_counts"].values()) == 3
    assert metadata["simulation"]["assignment_policy"] == "dirichlet_sampled"
    assert metadata["simulation"]["sampled_client_persona"] == metadata["persona_assignment"]["sampled_client_persona"]
    assert metadata["simulation"]["persona"]["persona"] == metadata["persona_assignment"]["sampled_client_persona"]


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_label_recommender_context_rejects_runs_without_generated_pairs(tmp_path) -> None:
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

    context_dir = run_dir / "clients" / "client_000" / "recommender_context" / selection_id
    context_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "client_id": ["client_000", "client_000"],
            "dataset_index": [0, 1],
            "instance_id": ["row-0", "row-1"],
            "method_variant": ["a", "a"],
            "metric_quality_z": [0.2, 0.5],
        }
    ).to_parquet(context_dir / "candidate_context.parquet", index=False)

    with pytest.raises(FileNotFoundError, match="No recommender preference pairs were generated"):
        label_recommender_context(
            run_id=run_id,
            selection_id=selection_id,
            persona="unit",
            persona_config_path=persona_path,
            concentration_c=1.0,
            paths=paths,
        )

    manifest_path = run_dir / "recommender_labels" / selection_id / "unit" / "labeling_manifest.json"
    assert not manifest_path.exists()


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_label_recommender_context_skips_clients_without_generated_pairs(tmp_path) -> None:
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

    usable_context_dir = run_dir / "clients" / "client_000" / "recommender_context" / selection_id
    usable_context_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "client_id": ["client_000"] * 6,
            "dataset_index": [0, 0, 0, 1, 1, 1],
            "instance_id": ["row-0"] * 3 + ["row-1"] * 3,
            "method_variant": ["a", "b", "c", "a", "b", "c"],
            "metric_quality_z": [0.2, 0.0, -0.2, 0.5, 0.1, -0.4],
        }
    ).to_parquet(usable_context_dir / "candidate_context.parquet", index=False)

    empty_context_dir = run_dir / "clients" / "client_001" / "recommender_context" / selection_id
    empty_context_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "client_id": ["client_001", "client_001"],
            "dataset_index": [0, 1],
            "instance_id": ["row-0", "row-1"],
            "method_variant": ["a", "a"],
            "metric_quality_z": [0.2, 0.5],
        }
    ).to_parquet(empty_context_dir / "candidate_context.parquet", index=False)

    summary = label_recommender_context(
        run_id=run_id,
        selection_id=selection_id,
        persona="unit",
        persona_config_path=persona_path,
        seed=42,
        label_seed=1729,
        concentration_c=1.0,
        paths=paths,
    )

    assert summary["status"] == "labeled"
    assert summary["client_count"] == 1
    assert summary["pair_count"] == 6
    assert [client["client_id"] for client in summary["clients"]] == ["client_000"]
    skipped_labels_path = (
        run_dir
        / "clients"
        / "client_001"
        / "recommender_labels"
        / selection_id
        / "unit"
        / "pairwise_labels.parquet"
    )
    assert not skipped_labels_path.exists()
