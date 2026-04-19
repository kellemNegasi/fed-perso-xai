"""Unified centralized and federated training orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fed_perso_xai.data.serialization import (
    ArraySplit,
    copy_shared_artifacts,
    load_array_split,
    load_client_datasets,
)
from fed_perso_xai.evaluation.comparison import build_baseline_comparison, write_comparison_report
from fed_perso_xai.evaluation.contracts import (
    ExtensionEvaluationBundle,
    PredictiveEvaluationBundle,
    SplitEvaluationReport,
)
from fed_perso_xai.evaluation.metrics import (
    compute_pooled_classification_metrics,
    summarize_class_balance,
    summarize_probability_distribution,
)
from fed_perso_xai.evaluation.predictions import build_prediction_artifact, save_prediction_artifact
from fed_perso_xai.models import TabularClassifier, create_model
from fed_perso_xai.utils.config import (
    CentralizedTrainingConfig,
    ComparisonConfig,
    FederatedTrainingConfig,
)
from fed_perso_xai.utils.paths import (
    centralized_run_dir,
    comparison_run_dir,
    federated_run_dir,
    partition_root,
    prepared_dir,
)
from fed_perso_xai.utils.provenance import (
    build_reproducibility_metadata,
    build_run_id,
    current_utc_timestamp,
    relative_artifact_path,
    resolve_git_commit_hash,
)


def train_centralized_from_prepared(
    config: CentralizedTrainingConfig,
) -> tuple[Path, dict[str, Any]]:
    """Train and evaluate the centralized baseline from prepared artifacts."""

    run_id = build_run_id(
        experiment_type="centralized",
        dataset_name=config.dataset_name,
        seed=config.seed,
    )
    prepared_root = prepared_dir(config.paths, config.dataset_name, config.seed)
    global_train = load_array_split(prepared_root / "global_train.npz")
    global_eval = load_array_split(prepared_root / "global_eval.npz")
    pooled_client_test = (
        load_array_split(prepared_root / "pooled_client_test.npz")
        if (prepared_root / "pooled_client_test.npz").exists()
        else None
    )

    model = create_model(
        config.model_name,
        n_features=global_train.X.shape[1],
        config=config.model,
    )
    train_loss = model.fit(global_train.X, global_train.y, seed=config.seed)

    result_dir = centralized_run_dir(config.paths, config.dataset_name, config.seed)
    result_dir.mkdir(parents=True, exist_ok=True)
    copy_shared_artifacts(prepared_root, result_dir)
    _copy_if_exists(prepared_root / "partition_metadata.json", result_dir / "partition_metadata.json")

    model_path = model.save(result_dir / "model_parameters.npz")
    config_path = result_dir / "config_snapshot.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    reproducibility_path = result_dir / "reproducibility_metadata.json"
    reproducibility_path.write_text(
        json.dumps(build_reproducibility_metadata(seed=config.seed), indent=2),
        encoding="utf-8",
    )

    global_eval_summary = _evaluate_split(
        model=model,
        run_id=run_id,
        dataset_name=config.dataset_name,
        split_name="global_eval",
        split=global_eval,
        prediction_path=result_dir / "predictions_global_eval.npz",
        provenance={
            "source": "prepared_global_eval",
            "prepared_root": str(prepared_root),
        },
    )
    pooled_summary = None
    if config.evaluate_on_pooled_client_test and pooled_client_test is not None:
        pooled_summary = _evaluate_split(
            model=model,
            run_id=run_id,
            dataset_name=config.dataset_name,
            split_name="pooled_client_test",
            split=pooled_client_test,
            prediction_path=result_dir / "predictions_pooled_client_test.npz",
            provenance={
                "source": "prepared_pooled_client_test",
                "prepared_root": str(prepared_root),
            },
        )

    predictive_evaluation = PredictiveEvaluationBundle(
        splits={
            "global_eval": global_eval_summary,
            **(
                {}
                if pooled_summary is None
                else {"pooled_client_test": pooled_summary}
            ),
        }
    )
    summary = {
        "run_id": run_id,
        "experiment_type": "centralized",
        "mode": "centralized",
        "dataset_name": config.dataset_name,
        "result_dir": str(result_dir),
        "config": config.to_dict(),
        "train_loss": float(train_loss),
        "evaluation": {
            "predictive": predictive_evaluation.to_dict(),
            "extensions": ExtensionEvaluationBundle().to_dict(),
        },
        "model_path": str(model_path),
    }
    metrics_path = result_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = _build_run_manifest(
        root_dir=result_dir,
        run_id=run_id,
        mode="centralized",
        dataset_name=config.dataset_name,
        config=config.to_dict(),
        seed_values={"global_seed": config.seed},
        artifact_paths={
            "config_snapshot": config_path,
            "metrics": metrics_path,
            "model": model_path,
            "preprocessor": result_dir / "preprocessor.joblib",
            "feature_metadata": result_dir / "feature_metadata.json",
            "dataset_metadata": result_dir / "dataset_metadata.json",
            "split_metadata": result_dir / "split_metadata.json",
            "partition_metadata": result_dir / "partition_metadata.json",
            "reproducibility_metadata": reproducibility_path,
            "predictions_global_eval": result_dir / "predictions_global_eval.npz",
            "predictions_pooled_client_test": result_dir / "predictions_pooled_client_test.npz",
        },
    )
    manifest_path = result_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary["run_manifest_path"] = str(manifest_path)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result_dir, summary


def train_federated_from_prepared(
    config: FederatedTrainingConfig,
) -> tuple[Any, dict[str, Any]]:
    """Load saved client arrays, run federated training, and persist outputs."""

    try:
        from fed_perso_xai.fl.client import ClientData
        from fed_perso_xai.fl.simulation import require_flower_support, run_federated_training
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "Federated training requires Flower support. Install `fed-perso-xai[fl]` "
            "for the debug runtime or `fed-perso-xai[ray]` for Ray-backed simulation."
        ) from exc
    try:
        require_flower_support()
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "Federated training requires Flower support. Install `fed-perso-xai[fl]` "
            "for the debug runtime or `fed-perso-xai[ray]` for Ray-backed simulation."
        ) from exc

    run_id = build_run_id(
        experiment_type="federated",
        dataset_name=config.dataset_name,
        seed=config.seed,
        num_clients=config.num_clients,
        alpha=config.alpha,
    )
    data_root = partition_root(
        config.paths.partition_root,
        config.dataset_name,
        config.num_clients,
        config.alpha,
        config.seed,
    )
    partition_metadata_path = _resolve_partition_metadata_path(data_root)
    if not partition_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing prepared partition metadata at '{partition_metadata_path}'. Run prepare-data first."
        )

    metadata = json.loads(partition_metadata_path.read_text(encoding="utf-8"))
    _validate_partition_metadata(metadata=metadata, config=config, metadata_path=partition_metadata_path)

    prepared_root = Path(metadata["prepared_root"])
    client_datasets = [
        ClientData(
            client_id=client.client_id,
            X_train=client.train.X,
            y_train=client.train.y,
            row_ids_train=client.train.row_ids,
            X_test=client.test.X,
            y_test=client.test.y,
            row_ids_test=client.test.row_ids,
        )
        for client in load_client_datasets(data_root, config.num_clients)
    ]
    result_dir = federated_run_dir(
        config.paths,
        config.dataset_name,
        config.num_clients,
        config.alpha,
        config.seed,
    )
    copy_shared_artifacts(prepared_root, result_dir)
    shutil.copy2(partition_metadata_path, result_dir / "partition_metadata.json")
    config_path = result_dir / "config_snapshot.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    reproducibility_path = result_dir / "reproducibility_metadata.json"
    reproducibility_path.write_text(
        json.dumps(build_reproducibility_metadata(seed=config.seed), indent=2),
        encoding="utf-8",
    )

    artifacts, summary = run_federated_training(
        client_datasets=client_datasets,
        config=config,
        result_dir=result_dir,
        run_id=run_id,
    )
    summary["run_id"] = run_id
    summary["mode"] = "federated"
    summary["run_manifest_path"] = str(result_dir / "run_manifest.json")
    metrics_path = artifacts.metrics_path
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = _build_run_manifest(
        root_dir=result_dir,
        run_id=run_id,
        mode="federated",
        dataset_name=config.dataset_name,
        config=config.to_dict(),
        seed_values={"global_seed": config.seed},
        artifact_paths={
            "config_snapshot": config_path,
            "metrics": metrics_path,
            "model": artifacts.model_path,
            "runtime_report": artifacts.runtime_path,
            "preprocessor": result_dir / "preprocessor.joblib",
            "feature_metadata": result_dir / "feature_metadata.json",
            "dataset_metadata": result_dir / "dataset_metadata.json",
            "split_metadata": result_dir / "split_metadata.json",
            "partition_metadata": result_dir / "partition_metadata.json",
            "reproducibility_metadata": reproducibility_path,
            "predictions_client_test": artifacts.predictions_path,
        },
    )
    manifest_path = result_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return artifacts, summary


def compare_centralized_and_federated(
    config: ComparisonConfig,
) -> tuple[Path, dict[str, Any]]:
    """Load centralized and federated outputs and persist a comparison report."""

    centralized_metrics_path = (
        centralized_run_dir(config.paths, config.dataset_name, config.seed) / "metrics_summary.json"
    )
    federated_metrics_path = federated_run_dir(
        config.paths,
        config.dataset_name,
        config.num_clients,
        config.alpha,
        config.seed,
    ) / "metrics_summary.json"
    if not centralized_metrics_path.exists() or not federated_metrics_path.exists():
        raise FileNotFoundError(
            "Both centralized and federated metrics_summary.json files must exist before comparison."
        )

    centralized_summary = json.loads(centralized_metrics_path.read_text(encoding="utf-8"))
    federated_summary = json.loads(federated_metrics_path.read_text(encoding="utf-8"))
    centralized_manifest = _load_run_manifest(
        centralized_run_dir(config.paths, config.dataset_name, config.seed) / "run_manifest.json"
    )
    federated_manifest = _load_run_manifest(
        federated_run_dir(
            config.paths,
            config.dataset_name,
            config.num_clients,
            config.alpha,
            config.seed,
        )
        / "run_manifest.json"
    )
    report = build_baseline_comparison(
        centralized_summary=centralized_summary,
        federated_summary=federated_summary,
        centralized_manifest=centralized_manifest,
        federated_manifest=federated_manifest,
    )
    report["config"] = config.to_dict()
    result_dir = comparison_run_dir(
        config.paths,
        config.dataset_name,
        config.num_clients,
        config.alpha,
        config.seed,
    )
    report_path = write_comparison_report(result_dir / "comparison_report.json", report)
    return report_path, report


def _evaluate_split(
    *,
    model: TabularClassifier,
    run_id: str,
    dataset_name: str,
    split_name: str,
    split: ArraySplit,
    prediction_path: Path,
    provenance: dict[str, Any],
) -> SplitEvaluationReport:
    probabilities = model.predict_proba(split.X)
    loss = model.loss(split.X, split.y)
    metrics = compute_pooled_classification_metrics(split.y, probabilities, loss=loss)
    predictions = build_prediction_artifact(
        run_id=run_id,
        dataset_name=dataset_name,
        split_name=split_name,
        y_true=split.y,
        y_prob=probabilities,
        row_ids=split.row_ids,
    )
    save_prediction_artifact(prediction_path, predictions)
    return SplitEvaluationReport(
        split_name=split_name,
        provenance=provenance,
        class_balance=summarize_class_balance(split.y),
        probability_summary=summarize_probability_distribution(probabilities),
        metrics=metrics,
        predictions_path=str(prediction_path),
    )


def _resolve_partition_metadata_path(root_dir: Path) -> Path:
    explicit = root_dir / "partition_metadata.json"
    if explicit.exists():
        return explicit
    return root_dir / "metadata.json"


def _validate_partition_metadata(
    *,
    metadata: dict[str, Any],
    config: FederatedTrainingConfig,
    metadata_path: Path,
) -> None:
    expected_values = {
        "dataset_name": config.dataset_name,
        "seed": config.seed,
        "num_clients": config.num_clients,
        "alpha": config.alpha,
    }
    mismatches = []
    for field_name, expected in expected_values.items():
        actual = metadata.get(field_name)
        if actual != expected:
            mismatches.append(
                f"{field_name}: expected {expected!r}, found {actual!r}"
            )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            f"Prepared partition metadata at '{metadata_path}' does not match the requested run: {details}."
        )


def _build_run_manifest(
    *,
    root_dir: Path,
    run_id: str,
    mode: str,
    dataset_name: str,
    config: dict[str, Any],
    seed_values: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    predictions = {
        name: relative_artifact_path(path, root_dir)
        for name, path in artifact_paths.items()
        if name.startswith("predictions_") and path.exists()
    }
    metadata_paths = {
        name: relative_artifact_path(path, root_dir)
        for name, path in artifact_paths.items()
        if name.endswith("_metadata") and path.exists()
    }
    return {
        "manifest_version": "stage1_run_manifest_v2",
        "run_id": run_id,
        "mode": mode,
        "dataset_name": dataset_name,
        "timestamp": current_utc_timestamp(),
        "git_commit_hash": resolve_git_commit_hash(root_dir),
        "seed_values": seed_values,
        "important_config": _extract_important_config_values(mode=mode, config=config),
        "artifacts": {
            "config_snapshot": _relative_if_exists(artifact_paths.get("config_snapshot"), root_dir),
            "model_artifact": _relative_if_exists(artifact_paths.get("model"), root_dir),
            "preprocessor_artifact": _relative_if_exists(artifact_paths.get("preprocessor"), root_dir),
            "feature_metadata_artifact": _relative_if_exists(
                artifact_paths.get("feature_metadata"),
                root_dir,
            ),
            "metrics_artifact": _relative_if_exists(artifact_paths.get("metrics"), root_dir),
            "predictions_artifacts": predictions,
            "metadata_artifacts": metadata_paths,
            "additional_artifacts": {
                name: relative_artifact_path(path, root_dir)
                for name, path in artifact_paths.items()
                if name not in {
                    "config_snapshot",
                    "model",
                    "preprocessor",
                    "feature_metadata",
                    "metrics",
                    *predictions.keys(),
                    *metadata_paths.keys(),
                }
                and path.exists()
            },
        },
    }


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _relative_if_exists(path: Path | None, root_dir: Path) -> str | None:
    if path is None or not path.exists():
        return None
    return relative_artifact_path(path, root_dir)


def _extract_important_config_values(*, mode: str, config: dict[str, Any]) -> dict[str, Any]:
    important = {
        "seed": config["seed"],
        "dataset_name": config["dataset_name"],
        "model_name": config["model_name"],
        "model": config["model"],
    }
    if mode == "centralized":
        important["evaluate_on_pooled_client_test"] = config["evaluate_on_pooled_client_test"]
        return important

    important.update(
        {
            "num_clients": config["num_clients"],
            "alpha": config["alpha"],
            "strategy_name": config["strategy_name"],
            "rounds": config["rounds"],
            "fit_fraction": config["fit_fraction"],
            "evaluate_fraction": config["evaluate_fraction"],
            "min_available_clients": config["min_available_clients"],
            "simulation_backend": config["simulation_backend"],
            "debug_fallback_on_error": config["debug_fallback_on_error"],
        }
    )
    return important


def _load_run_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
