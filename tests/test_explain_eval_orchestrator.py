from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

from fed_perso_xai.data.serialization import ArraySplit, save_federated_dataset
from fed_perso_xai.models import create_model, save_global_model_parameters
from fed_perso_xai.orchestration.explain_eval import run_explain_eval_job
from fed_perso_xai.utils.config import ArtifactPaths, LogisticRegressionConfig
from fed_perso_xai.utils.paths import federated_run_artifact_dir, federated_run_metadata_path

PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


def _build_paths(tmp_path: Path) -> ArtifactPaths:
    return ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )


def _materialize_run_artifact(tmp_path: Path, synthetic_client_splits) -> tuple[ArtifactPaths, str]:
    paths = _build_paths(tmp_path)
    prepared_root = paths.prepared_root / "toy" / "seed_7"
    prepared_root.mkdir(parents=True, exist_ok=True)
    feature_metadata_path = prepared_root / "feature_metadata.json"
    feature_metadata_path.write_text(
        json.dumps(
            {
                "stable_transformed_feature_order": [f"feature_{idx}" for idx in range(4)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (prepared_root / "preprocessor.joblib").write_text("stub", encoding="utf-8")

    client_splits = []
    stacked_train_X = []
    stacked_train_y = []
    for split in synthetic_client_splits:
        client_splits.append(
            type(
                "ClientSplitStub",
                (),
                {
                    "client_id": split["client_id"],
                    "X_train": split["X_train"],
                    "y_train": split["y_train"],
                    "row_ids_train": split["row_ids_train"],
                    "X_test": split["X_test"],
                    "y_test": split["y_test"],
                    "row_ids_test": split["row_ids_test"],
                    "train_size": split["X_train"].shape[0],
                    "test_size": split["X_test"].shape[0],
                },
            )()
        )
        stacked_train_X.append(split["X_train"])
        stacked_train_y.append(split["y_train"])

    dataset_artifacts = save_federated_dataset(
        dataset_name="toy",
        output_root=paths.partition_root,
        num_clients=3,
        alpha=1.0,
        seed=7,
        prepared_root=prepared_root,
        preprocessor_path=prepared_root / "preprocessor.joblib",
        feature_metadata_path=feature_metadata_path,
        client_splits=client_splits,
    )

    model = create_model(
        "logistic_regression",
        n_features=4,
        config=LogisticRegressionConfig(epochs=6, batch_size=4, learning_rate=0.1),
    )
    model.fit(np.vstack(stacked_train_X), np.concatenate(stacked_train_y), seed=13)

    run_id = "federated-training-toy-20260423t120000z-logistic_regression-3clients-alpha1.0-seed7-demo"
    run_root = federated_run_artifact_dir(paths, run_id)
    model_path = save_global_model_parameters(run_root / "model" / "global_model.npz", model)
    model_metadata = {
        "run_id": run_id,
        "model_type": "logistic_regression",
        "model_config": LogisticRegressionConfig(epochs=6, batch_size=4, learning_rate=0.1).__dict__,
        "n_features": 4,
        "parameter_count": len(model.get_parameters()),
        "class_labels": [0, 1],
        "serialization_format": "numpy_parameter_bundle_npz_v1",
        "model_artifact_path": "model/global_model.npz",
        "training_metadata_path": "training/training_metadata.json",
    }
    (run_root / "model").mkdir(parents=True, exist_ok=True)
    (run_root / "training").mkdir(parents=True, exist_ok=True)
    (run_root / "model" / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2),
        encoding="utf-8",
    )
    training_metadata = {
        "status": "completed",
        "run_id": run_id,
        "dataset_name": "toy",
        "partition_data_root": str(dataset_artifacts.root_dir.resolve()),
        "partition_metadata_path": str(dataset_artifacts.partition_metadata_path),
        "partition_metadata_sha256": "demo",
        "prepared_root": str(prepared_root),
        "num_clients": 3,
        "alpha": 1.0,
        "model_type": "logistic_regression",
        "training_config": {
            "dataset_name": "toy",
            "seed": 7,
            "model_name": "logistic_regression",
            "num_clients": 3,
            "alpha": 1.0,
        },
        "training_config_sha256": "demo-sha",
        "seed_values": {"global_seed": 7, "secure_seed": 0},
        "started_at": "2026-04-23T12:00:00+00:00",
        "completed_at": "2026-04-23T12:01:00+00:00",
    }
    (run_root / "training" / "training_metadata.json").write_text(
        json.dumps(training_metadata, indent=2),
        encoding="utf-8",
    )
    (run_root / "feature_metadata.json").write_text(
        feature_metadata_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    federated_run_metadata_path(run_root).write_text(
        json.dumps(
            {
                "artifact_type": "federated_run",
                "artifact_version": "federated_run_v2",
                "run_id": run_id,
                "created_at": training_metadata["started_at"],
                "completed_at": training_metadata["completed_at"],
                "status": "completed",
                "model_type": "logistic_regression",
                "training_config": training_metadata["training_config"],
                "training_config_sha256": training_metadata["training_config_sha256"],
                "seed_values": training_metadata["seed_values"],
                "model_artifact_path": "model/global_model.npz",
                "model_metadata_path": "model/model_metadata.json",
                "training_metadata_path": "training/training_metadata.json",
                "feature_metadata_path": "feature_metadata.json",
                "partition_reference": {
                    "partition_data_root": str(dataset_artifacts.root_dir.resolve()),
                    "partition_metadata_path": str(dataset_artifacts.partition_metadata_path),
                    "partition_metadata_sha256": "demo",
                    "feature_metadata_path": str(feature_metadata_path),
                    "prepared_root": str(prepared_root),
                    "num_clients": 3,
                    "alpha": 1.0,
                },
                "model_summary": {
                    "n_features": 4,
                    "parameter_count": len(model.get_parameters()),
                    "serialization_format": "numpy_parameter_bundle_npz_v1",
                    "class_labels": [0, 1],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert model_path.exists()
    return paths, run_id


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_explain_eval_job_writes_parquet_json_and_metadata(tmp_path, synthetic_client_splits) -> None:
    import pyarrow.parquet as pq

    paths, run_id = _materialize_run_artifact(tmp_path, synthetic_client_splits)
    payload = run_explain_eval_job(
        run_id=run_id,
        client_id="client_000",
        split="test",
        shard_id="shard_000",
        explainer_name="lime",
        config_id="lime__kernel-1.5__samples-50",
        max_instances=4,
        random_state=42,
        paths=paths,
        metric_names=["compactness_size"],
    )

    explanations_path = Path(payload["artifacts"]["detailed_explanations_path"])
    metrics_path = Path(payload["artifacts"]["metrics_results_path"])
    job_metadata_path = Path(payload["artifacts"]["job_metadata_path"])
    done_path = Path(payload["artifacts"]["done_marker_path"])
    shard_metadata_path = Path(payload["artifacts"]["shard_metadata_path"])
    client_metadata_path = Path(payload["artifacts"]["client_metadata_path"])

    assert explanations_path.suffix == ".parquet"
    assert explanations_path.exists()
    assert metrics_path.exists()
    assert job_metadata_path.exists()
    assert shard_metadata_path.exists()
    assert client_metadata_path.exists()
    assert done_path.exists()

    table = pq.read_table(explanations_path)
    assert table.num_rows == 4
    assert set(table.column_names) >= {
        "run_id",
        "client_id",
        "split",
        "shard_id",
        "explainer_name",
        "config_id",
        "instance_id",
        "dataset_index",
        "attributions_json",
        "explanation_metadata_json",
    }

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["run_id"] == run_id
    assert metrics_payload["client_id"] == "client_000"
    assert metrics_payload["config_id"] == "lime__kernel-1.5__samples-50"
    assert metrics_payload["metric_names"] == ["compactness_size"]
    assert len(metrics_payload["per_instance_results"]) == 4


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_sampling_is_stable_and_changes_with_seed(tmp_path, synthetic_client_splits) -> None:
    paths, run_id = _materialize_run_artifact(tmp_path, synthetic_client_splits)
    job_a = run_explain_eval_job(
        run_id=run_id,
        client_id="client_000",
        explainer_name="lime",
        config_id="lime__kernel-1.5__samples-50",
        max_instances=3,
        random_state=11,
        paths=paths,
        metric_names=["compactness_size"],
    )
    first_indices = job_a["selected_dataset_indices"]

    job_b = run_explain_eval_job(
        run_id=run_id,
        client_id="client_000",
        explainer_name="lime",
        config_id="lime__kernel-2.0__samples-50",
        max_instances=3,
        random_state=11,
        paths=paths,
        metric_names=["compactness_size"],
    )
    assert job_b["selected_dataset_indices"] == first_indices

    job_c = run_explain_eval_job(
        run_id=run_id,
        client_id="client_000",
        explainer_name="lime",
        config_id="lime__kernel-3.0__samples-50",
        max_instances=3,
        random_state=99,
        paths=paths,
        metric_names=["compactness_size"],
    )
    assert job_c["selected_dataset_indices"] != first_indices


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_rerun_skips_and_force_overrides(tmp_path, synthetic_client_splits) -> None:
    paths, run_id = _materialize_run_artifact(tmp_path, synthetic_client_splits)
    first = run_explain_eval_job(
        run_id=run_id,
        client_id="client_000",
        explainer_name="lime",
        config_id="lime__kernel-1.5__samples-100",
        max_instances=2,
        random_state=7,
        paths=paths,
        metric_names=["compactness_size"],
    )
    second = run_explain_eval_job(
        run_id=run_id,
        client_id="client_000",
        explainer_name="lime",
        config_id="lime__kernel-1.5__samples-100",
        max_instances=2,
        random_state=7,
        paths=paths,
        metric_names=["compactness_size"],
    )
    assert second["status"] == "skipped_existing"
    time.sleep(0.01)
    third = run_explain_eval_job(
        run_id=run_id,
        client_id="client_000",
        explainer_name="lime",
        config_id="lime__kernel-1.5__samples-100",
        max_instances=2,
        random_state=7,
        force=True,
        paths=paths,
        metric_names=["compactness_size"],
    )
    assert third["status"] == "completed"
    assert third["generated_at"] != first["generated_at"]


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_done_marker_is_not_written_on_failure(tmp_path, synthetic_client_splits, monkeypatch) -> None:
    from fed_perso_xai.orchestration import explain_eval as module

    paths, run_id = _materialize_run_artifact(tmp_path, synthetic_client_splits)
    original_write_json_atomic = module._write_json_atomic

    def failing_write(path: Path, payload: dict[str, object]) -> None:
        if path.name.endswith(".json") and "metrics_results" in str(path):
            raise RuntimeError("forced metrics write failure")
        return original_write_json_atomic(path, payload)

    monkeypatch.setattr(module, "_write_json_atomic", failing_write)
    with pytest.raises(RuntimeError, match="forced metrics write failure"):
        run_explain_eval_job(
            run_id=run_id,
            client_id="client_000",
            explainer_name="lime",
            config_id="lime__kernel-2.0__samples-100",
            max_instances=2,
            random_state=5,
            paths=paths,
            metric_names=["compactness_size"],
        )

    shard_root = paths.federated_root / "runs" / run_id / "clients" / "client_000" / "test_shards" / "shard_000"
    assert not (shard_root / "_status" / "lime__kernel-2.0__samples-100.done").exists()


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet artifact tests.")
def test_independent_jobs_do_not_collide_on_paths(tmp_path, synthetic_client_splits) -> None:
    paths, run_id = _materialize_run_artifact(tmp_path, synthetic_client_splits)
    job_a = run_explain_eval_job(
        run_id=run_id,
        client_id="client_000",
        explainer_name="lime",
        config_id="lime__kernel-1.5__samples-50",
        max_instances=2,
        random_state=1,
        paths=paths,
        metric_names=["compactness_size"],
    )
    job_b = run_explain_eval_job(
        run_id=run_id,
        client_id="client_001",
        explainer_name="lime",
        config_id="lime__kernel-2.0__samples-100",
        max_instances=2,
        random_state=1,
        paths=paths,
        metric_names=["compactness_size"],
    )
    assert job_a["artifacts"]["detailed_explanations_path"] != job_b["artifacts"]["detailed_explanations_path"]
    assert job_a["artifacts"]["metrics_results_path"] != job_b["artifacts"]["metrics_results_path"]
    assert Path(job_a["artifacts"]["detailed_explanations_path"]).exists()
    assert Path(job_b["artifacts"]["detailed_explanations_path"]).exists()
