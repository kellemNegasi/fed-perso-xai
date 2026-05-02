from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from fed_perso_xai.orchestration.recommender_training import (
    _recommender_training_dir,
    _resolve_recommender_training_dir,
    evaluate_recommender_model,
    train_federated_recommender,
)
from fed_perso_xai.recommender.evaluation import (
    evaluate_grouped_ranked_scores,
    evaluate_ranked_scores,
)
from fed_perso_xai.utils.config import ArtifactPaths, RecommenderFederatedTrainingConfig

FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None
PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


def _paths(tmp_path):
    return ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )


def test_recommender_training_dir_includes_training_variant() -> None:
    run_root = Path("/tmp/unit-run")

    assert _recommender_training_dir(
        run_root,
        "selection-0",
        "lay",
        secure_aggregation=False,
    ) == run_root / "recommender_training" / "selection-0" / "lay" / "svm_rank_fedavg" / "plain"
    assert _recommender_training_dir(
        run_root,
        "selection-0",
        "lay",
        secure_aggregation=True,
    ) == run_root / "recommender_training" / "selection-0" / "lay" / "svm_rank_fedavg" / "secure"
    assert _recommender_training_dir(
        run_root,
        "selection-0",
        "lay",
        secure_aggregation=False,
        clustered=True,
    ) == run_root / "recommender_training" / "selection-0" / "lay" / "svm_rank_fedavg" / "clustered"


def test_resolve_recommender_training_dir_requires_explicit_variant_when_multiple_exist(tmp_path) -> None:
    run_root = tmp_path / "run"
    plain_dir = _recommender_training_dir(
        run_root,
        "selection-0",
        "lay",
        secure_aggregation=False,
    )
    secure_dir = _recommender_training_dir(
        run_root,
        "selection-0",
        "lay",
        secure_aggregation=True,
    )
    plain_dir.mkdir(parents=True)
    secure_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="Multiple recommender training directories exist"):
        _resolve_recommender_training_dir(run_root, "selection-0", "lay")


def test_resolve_recommender_training_dir_prefers_clustered_variant_when_requested(tmp_path) -> None:
    run_root = tmp_path / "run"
    clustered_dir = _recommender_training_dir(
        run_root,
        "selection-0",
        "lay",
        secure_aggregation=False,
        clustered=True,
    )
    clustered_dir.mkdir(parents=True)

    resolved = _resolve_recommender_training_dir(
        run_root,
        "selection-0",
        "lay",
        clustered=True,
    )

    assert resolved == clustered_dir


def test_resolve_recommender_training_dir_falls_back_to_legacy_plain_dir(tmp_path) -> None:
    run_root = tmp_path / "run"
    legacy_dir = run_root / "recommender_training" / "selection-0" / "lay" / "svm_rank_fedavg"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "training_metadata.json").write_text(
        json.dumps({"config": {"secure_aggregation": False}}),
        encoding="utf-8",
    )

    resolved = _resolve_recommender_training_dir(
        run_root,
        "selection-0",
        "lay",
        secure_aggregation=False,
    )

    assert resolved == legacy_dir


@pytest.mark.parametrize(
    ("selection_id", "persona", "expected_label"),
    [
        ("../escape", "lay", "selection_id"),
        ("selection-0", "../other", "persona"),
    ],
)
def test_evaluate_recommender_model_rejects_unsafe_path_segments(
    tmp_path,
    selection_id: str,
    persona: str,
    expected_label: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{expected_label} must be a single non-empty path segment\."):
        evaluate_recommender_model(
            run_id="unit-run",
            selection_id=selection_id,
            persona=persona,
            paths=_paths(tmp_path),
        )


@pytest.mark.parametrize(
    ("context_filename", "label_filename", "expected_label"),
    [
        ("../context.parquet", "pairwise_labels.parquet", "context_filename"),
        ("candidate_context.parquet", "../labels.parquet", "label_filename"),
        ("/tmp/context.parquet", "pairwise_labels.parquet", "context_filename"),
        ("candidate_context.parquet", "/tmp/labels.parquet", "label_filename"),
    ],
)
def test_evaluate_recommender_model_rejects_unsafe_filenames(
    tmp_path,
    context_filename: str,
    label_filename: str,
    expected_label: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{expected_label} must be a single non-empty path segment\."):
        evaluate_recommender_model(
            run_id="unit-run",
            selection_id="selection-0",
            persona="lay",
            context_filename=context_filename,
            label_filename=label_filename,
            paths=_paths(tmp_path),
        )


@pytest.mark.parametrize(
    ("context_filename", "label_filename", "expected_label"),
    [
        ("../context.parquet", "pairwise_labels.parquet", "context_filename"),
        ("candidate_context.parquet", "../labels.parquet", "label_filename"),
        ("/tmp/context.parquet", "pairwise_labels.parquet", "context_filename"),
        ("candidate_context.parquet", "/tmp/labels.parquet", "label_filename"),
    ],
)
def test_train_federated_recommender_rejects_unsafe_filenames(
    tmp_path,
    context_filename: str,
    label_filename: str,
    expected_label: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{expected_label} must be a single non-empty path segment\."):
        train_federated_recommender(
            RecommenderFederatedTrainingConfig(
                run_id="unit-run",
                selection_id="selection-0",
                persona="lay",
                paths=_paths(tmp_path),
                context_filename=context_filename,
                label_filename=label_filename,
            )
        )


def test_evaluate_ranked_scores_uses_global_pairwise_order_and_pearson() -> None:
    labels = pd.DataFrame(
        {
            "dataset_index": [0, 1, 2],
            "pair_1": ["a", "a", "a"],
            "pair_2": ["b", "b", "b"],
            "label": [0, 0, 0],
        }
    )

    metrics = evaluate_ranked_scores(
        predicted_scores={"a": 2.0, "b": -1.0},
        pair_labels=labels,
        top_k=(1, 2, 5, 8),
    )

    assert metrics["ground_truth_order"] == ["a", "b"]
    assert metrics["predicted_order"] == ["a", "b"]
    assert metrics["precision_at_1"] == pytest.approx(1.0)
    assert metrics["precision_at_2"] == pytest.approx(1.0)
    assert metrics["precision_at_5"] == pytest.approx(1.0)
    assert metrics["precision_at_8"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["pearson_at_1"] == pytest.approx(1.0)
    assert metrics["pearson_at_2"] == pytest.approx(1.0)
    assert metrics["pearson_at_5"] == pytest.approx(1.0)
    assert metrics["pearson_at_8"] == pytest.approx(1.0)


def test_evaluate_grouped_ranked_scores_keeps_instances_separate() -> None:
    candidate_scores = pd.DataFrame(
        {
            "dataset_index": [0, 0, 1, 1],
            "method_variant": ["a", "b", "a", "b"],
            "score": [2.0, -1.0, -2.0, 1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "dataset_index": [0, 1],
            "pair_1": ["a", "a"],
            "pair_2": ["b", "b"],
            "label": [0, 1],
        }
    )

    metrics = evaluate_grouped_ranked_scores(
        candidate_scores=candidate_scores,
        pair_labels=labels,
        top_k=(1, 2, 8),
    )

    assert metrics["instance_count"] == 2
    assert metrics["aggregate"]["precision_at_1"] == pytest.approx(1.0)
    assert metrics["aggregate"]["precision_at_8"] == pytest.approx(1.0)
    assert metrics["aggregate"]["pearson"] == pytest.approx(1.0)
    assert metrics["aggregate"]["pearson_at_1"] == pytest.approx(1.0)
    assert metrics["aggregate"]["pearson_at_8"] == pytest.approx(1.0)
    assert "dataset_index" not in metrics["aggregate"]


@pytest.mark.skipif(not FLOWER_AVAILABLE, reason="Flower is required for recommender FL tests.")
@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_train_federated_recommender_allows_clients_without_eval_pairs(tmp_path) -> None:
    paths = _paths(tmp_path)
    run_id = "unit-run"
    selection = "test__max-2__seed-9"
    persona = "lay"
    run_dir = paths.federated_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    for client_idx in range(2):
        client_dir = run_dir / "clients" / f"client_{client_idx:03d}"
        context_dir = client_dir / "recommender_context" / selection
        label_dir = client_dir / "recommender_labels" / selection / persona
        context_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        candidates = pd.DataFrame(
            {
                "client_id": [f"client_{client_idx:03d}"] * 4,
                "dataset_index": [0, 0, 1, 1],
                "instance_id": ["i0", "i0", "i1", "i1"],
                "method_variant": ["a", "b", "a", "b"],
                "metric_quality_z": [2.0, -2.0, 1.5, -1.5],
                "candidate_index_within_instance": [0, 1, 0, 1],
            }
        )
        if client_idx == 0:
            labels = pd.DataFrame(
                {
                    "client_id": [f"client_{client_idx:03d}"] * 2,
                    "dataset_index": [0, 1],
                    "pair_1": ["a", "a"],
                    "pair_2": ["b", "b"],
                    "label": [0, 0],
                    "split": ["train", "test"],
                }
            )
        else:
            labels = pd.DataFrame(
                {
                    "client_id": [f"client_{client_idx:03d}"],
                    "dataset_index": [0],
                    "pair_1": ["a"],
                    "pair_2": ["b"],
                    "label": [0],
                    "split": ["train"],
                }
            )
        candidates.to_parquet(context_dir / "candidate_context.parquet", index=False)
        labels.to_parquet(label_dir / "pairwise_labels.parquet", index=False)
        (label_dir / "simulation_metadata.json").write_text(
            json.dumps(
                {
                    "instance_split": {
                        "train_dataset_indices": [0],
                        "test_dataset_indices": [1],
                    }
                }
            ),
            encoding="utf-8",
        )

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            paths=paths,
            rounds=2,
            epochs=5,
            batch_size=2,
            learning_rate=0.2,
            simulation_backend="debug-sequential",
            min_available_clients=2,
            top_k=(1, 2),
        )
    )

    assert artifacts.model_artifact_path.exists()
    assert artifacts.run_dir.name == "plain"
    assert metadata["status"] == "completed"
    assert metadata["clients_without_eval"] == ["client_001"]
    assert metadata["raw_pair_count"] == 2
    assert metadata["eval_raw_pair_count"] == 1

    evaluation = json.loads(artifacts.evaluation_summary_path.read_text(encoding="utf-8"))
    assert evaluation["client_count"] == 1


@pytest.mark.skipif(not FLOWER_AVAILABLE, reason="Flower is required for recommender FL tests.")
@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_train_federated_recommender_skips_final_evaluation_when_no_test_pairs_exist(tmp_path) -> None:
    paths = _paths(tmp_path)
    run_id = "unit-run"
    selection = "test__max-2__seed-9"
    persona = "lay"
    run_dir = paths.federated_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    for client_idx in range(2):
        client_dir = run_dir / "clients" / f"client_{client_idx:03d}"
        context_dir = client_dir / "recommender_context" / selection
        label_dir = client_dir / "recommender_labels" / selection / persona
        context_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        candidates = pd.DataFrame(
            {
                "client_id": [f"client_{client_idx:03d}"] * 4,
                "dataset_index": [0, 0, 1, 1],
                "instance_id": ["i0", "i0", "i1", "i1"],
                "method_variant": ["a", "b", "a", "b"],
                "metric_quality_z": [2.0, -2.0, 1.5, -1.5],
                "candidate_index_within_instance": [0, 1, 0, 1],
            }
        )
        labels = pd.DataFrame(
            {
                "client_id": [f"client_{client_idx:03d}"],
                "dataset_index": [0],
                "pair_1": ["a"],
                "pair_2": ["b"],
                "label": [0],
                "split": ["train"],
            }
        )
        candidates.to_parquet(context_dir / "candidate_context.parquet", index=False)
        labels.to_parquet(label_dir / "pairwise_labels.parquet", index=False)
        (label_dir / "simulation_metadata.json").write_text(
            json.dumps(
                {
                    "instance_split": {
                        "train_dataset_indices": [0],
                        "test_dataset_indices": [1],
                    }
                }
            ),
            encoding="utf-8",
        )

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            paths=paths,
            rounds=2,
            epochs=5,
            batch_size=2,
            learning_rate=0.2,
            simulation_backend="debug-sequential",
            min_available_clients=2,
            top_k=(1, 2),
        )
    )

    evaluation = json.loads(artifacts.evaluation_summary_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["clients_without_eval"] == ["client_000", "client_001"]
    assert evaluation["status"] == "skipped_no_test_pairs"
    assert evaluation["client_count"] == 0


@pytest.mark.skipif(not FLOWER_AVAILABLE, reason="Flower is required for recommender FL tests.")
@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_train_federated_recommender_writes_model_metadata_and_evaluation(tmp_path) -> None:
    paths = _paths(tmp_path)
    run_id = "unit-run"
    selection = "test__max-2__seed-9"
    persona = "lay"
    run_dir = paths.federated_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    for client_idx in range(2):
        client_dir = run_dir / "clients" / f"client_{client_idx:03d}"
        context_dir = client_dir / "recommender_context" / selection
        label_dir = client_dir / "recommender_labels" / selection / persona
        context_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        candidates = pd.DataFrame(
            {
                "client_id": [f"client_{client_idx:03d}"] * 4,
                "dataset_index": [0, 0, 1, 1],
                "instance_id": ["i0", "i0", "i1", "i1"],
                "method_variant": ["a", "b", "a", "b"],
                "metric_quality_z": [2.0, -2.0, 1.5, -1.5],
                "candidate_index_within_instance": [0, 1, 0, 1],
            }
        )
        labels = pd.DataFrame(
            {
                "client_id": [f"client_{client_idx:03d}"] * 2,
                "dataset_index": [0, 1],
                "pair_1": ["a", "a"],
                "pair_2": ["b", "b"],
                "label": [0, 0],
                "split": ["train", "test"],
            }
        )
        candidates.to_parquet(context_dir / "candidate_context.parquet", index=False)
        labels.to_parquet(label_dir / "pairwise_labels.parquet", index=False)
        (label_dir / "simulation_metadata.json").write_text(
            json.dumps(
                {
                    "instance_split": {
                        "train_dataset_indices": [0],
                        "test_dataset_indices": [1],
                    }
                }
            ),
            encoding="utf-8",
        )

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            paths=paths,
            rounds=2,
            epochs=5,
            batch_size=2,
            learning_rate=0.2,
            simulation_backend="debug-sequential",
            min_available_clients=2,
            top_k=(1, 2),
        )
    )

    assert artifacts.model_artifact_path.exists()
    assert artifacts.model_metadata_path.exists()
    assert artifacts.feature_metadata_path.exists()
    assert artifacts.evaluation_summary_path.exists()
    assert metadata["status"] == "completed"
    assert metadata["recommender_type"] == "svm_rank"
    assert metadata["feature_columns"] == ["metric_quality_z"]
    assert metadata["raw_pair_count"] == 2
    assert metadata["eval_raw_pair_count"] == 2

    model_metadata = json.loads(artifacts.model_metadata_path.read_text(encoding="utf-8"))
    assert model_metadata["recommender_type"] == "svm_rank"
    assert model_metadata["model_type"] == "svm_rank_recommender"

    evaluation = json.loads(artifacts.evaluation_summary_path.read_text(encoding="utf-8"))
    assert evaluation["aggregate"]["precision_at_1"] == pytest.approx(1.0)
    assert evaluation["aggregate"]["pearson"] == pytest.approx(1.0)
    assert "dataset_index" not in evaluation["aggregate"]
    assert "dataset_index" not in metadata["evaluation"]


@pytest.mark.skipif(not FLOWER_AVAILABLE, reason="Flower is required for recommender FL tests.")
@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_train_federated_recommender_can_skip_round_evaluation(tmp_path) -> None:
    paths = _paths(tmp_path)
    run_id = "unit-run"
    selection = "test__max-2__seed-9"
    persona = "lay"
    run_dir = paths.federated_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    for client_idx in range(2):
        client_dir = run_dir / "clients" / f"client_{client_idx:03d}"
        context_dir = client_dir / "recommender_context" / selection
        label_dir = client_dir / "recommender_labels" / selection / persona
        context_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        candidates = pd.DataFrame(
            {
                "client_id": [f"client_{client_idx:03d}"] * 4,
                "dataset_index": [0, 0, 1, 1],
                "instance_id": ["i0", "i0", "i1", "i1"],
                "method_variant": ["a", "b", "a", "b"],
                "metric_quality_z": [2.0, -2.0, 1.5, -1.5],
                "candidate_index_within_instance": [0, 1, 0, 1],
            }
        )
        labels = pd.DataFrame(
            {
                "client_id": [f"client_{client_idx:03d}"] * 2,
                "dataset_index": [0, 1],
                "pair_1": ["a", "a"],
                "pair_2": ["b", "b"],
                "label": [0, 0],
                "split": ["train", "test"],
            }
        )
        candidates.to_parquet(context_dir / "candidate_context.parquet", index=False)
        labels.to_parquet(label_dir / "pairwise_labels.parquet", index=False)
        (label_dir / "simulation_metadata.json").write_text(
            json.dumps(
                {
                    "instance_split": {
                        "train_dataset_indices": [0],
                        "test_dataset_indices": [1],
                    }
                }
            ),
            encoding="utf-8",
        )

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            paths=paths,
            rounds=2,
            epochs=5,
            batch_size=2,
            learning_rate=0.2,
            evaluate_fraction=0.0,
            simulation_backend="debug-sequential",
            min_available_clients=2,
            top_k=(1, 2),
        )
    )

    history = list(csv.DictReader(artifacts.training_history_path.open("r", encoding="utf-8")))
    assert history
    assert {row["evaluate_skipped"] for row in history} == {"True"}
    assert {row["evaluate_loss"] for row in history} == {""}
    evaluation = json.loads(artifacts.evaluation_summary_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert evaluation["aggregate"]["precision_at_1"] == pytest.approx(1.0)


@pytest.mark.skipif(not FLOWER_AVAILABLE, reason="Flower is required for recommender FL tests.")
@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_train_federated_recommender_supports_explicit_pairwise_logistic_selection(tmp_path) -> None:
    paths = _paths(tmp_path)
    run_id = "unit-run"
    selection = "test__max-2__seed-9"
    persona = "lay"
    run_dir = paths.federated_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    for client_idx in range(2):
        client_dir = run_dir / "clients" / f"client_{client_idx:03d}"
        context_dir = client_dir / "recommender_context" / selection
        label_dir = client_dir / "recommender_labels" / selection / persona
        context_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        candidates = pd.DataFrame(
            {
                "client_id": [f"client_{client_idx:03d}"] * 4,
                "dataset_index": [0, 0, 1, 1],
                "instance_id": ["i0", "i0", "i1", "i1"],
                "method_variant": ["a", "b", "a", "b"],
                "metric_quality_z": [2.0, -2.0, 1.5, -1.5],
                "candidate_index_within_instance": [0, 1, 0, 1],
            }
        )
        labels = pd.DataFrame(
            {
                "client_id": [f"client_{client_idx:03d}"] * 2,
                "dataset_index": [0, 1],
                "pair_1": ["a", "a"],
                "pair_2": ["b", "b"],
                "label": [0, 0],
                "split": ["train", "test"],
            }
        )
        candidates.to_parquet(context_dir / "candidate_context.parquet", index=False)
        labels.to_parquet(label_dir / "pairwise_labels.parquet", index=False)
        (label_dir / "simulation_metadata.json").write_text(
            json.dumps(
                {
                    "instance_split": {
                        "train_dataset_indices": [0],
                        "test_dataset_indices": [1],
                    }
                }
            ),
            encoding="utf-8",
        )

    artifacts, metadata = train_federated_recommender(
        RecommenderFederatedTrainingConfig(
            run_id=run_id,
            selection_id=selection,
            persona=persona,
            recommender_type="pairwise_logistic",
            paths=paths,
            rounds=2,
            epochs=5,
            batch_size=2,
            learning_rate=0.2,
            simulation_backend="debug-sequential",
            min_available_clients=2,
            top_k=(1, 2),
        )
    )

    assert artifacts.model_artifact_path.exists()
    assert metadata["status"] == "completed"
    assert metadata["recommender_type"] == "pairwise_logistic"

    model_metadata = json.loads(artifacts.model_metadata_path.read_text(encoding="utf-8"))
    assert model_metadata["recommender_type"] == "pairwise_logistic"
    assert model_metadata["model_type"] == "pairwise_logistic_recommender"

    evaluation = json.loads(artifacts.evaluation_summary_path.read_text(encoding="utf-8"))
    assert evaluation["aggregate"]["precision_at_1"] == pytest.approx(1.0)
    assert evaluation["aggregate"]["pearson"] == pytest.approx(1.0)
    assert "dataset_index" not in evaluation["aggregate"]
    assert "dataset_index" not in metadata["evaluation"]
