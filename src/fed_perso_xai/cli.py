"""CLI entrypoints for the baseline experiment workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fed_perso_xai.data.catalog import DEFAULT_DATASET_REGISTRY
from fed_perso_xai.fl.strategy import DEFAULT_STRATEGY_REGISTRY
from fed_perso_xai.models.registry import DEFAULT_MODEL_REGISTRY
from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.orchestration.explain_eval import (
    plan_explain_eval_jobs,
    run_explain_eval_job,
    run_explain_eval_plan_item,
)
from fed_perso_xai.orchestration.explain_eval_aggregation import (
    aggregate_explain_eval_results,
)
from fed_perso_xai.orchestration.explanations import (
    generate_client_local_explanations,
    load_client_data_for_explanations,
    load_saved_model_for_explanations,
    resolve_partition_data_root_for_explanations,
    resolve_feature_names_for_explanations,
    save_client_explanations,
)
from fed_perso_xai.orchestration.federated_training import train_federated_from_partitions
from fed_perso_xai.orchestration.job_launcher import run_job_launcher
from fed_perso_xai.orchestration.recommender_preprocessing import (
    prepare_recommender_context,
)
from fed_perso_xai.orchestration.recommender_training import (
    evaluate_recommender_model,
    train_federated_recommender,
)
from fed_perso_xai.orchestration.training import (
    compare_centralized_and_federated,
    train_centralized_from_prepared,
)
from fed_perso_xai.recommender import (
    DEFAULT_RECOMMENDER_TYPE,
    SUPPORTED_RECOMMENDER_TYPES,
    label_recommender_context,
)
from fed_perso_xai.utils.config import (
    ArtifactPaths,
    CentralizedTrainingConfig,
    ComparisonConfig,
    DataPreparationConfig,
    FederatedTrainingConfig,
    LogisticRegressionConfig,
    PartitionConfig,
    RecommenderClusteringConfig,
    RecommenderFederatedTrainingConfig,
    PreprocessingConfig,
)
from fed_perso_xai.utils.logging import configure_logging

FEDERATED_BACKEND_CHOICES = ["auto", "ray", "debug-sequential", "sequential_fallback"]
MODEL_SOURCE_CHOICES = ["federated", "centralized"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    dataset_choices = DEFAULT_DATASET_REGISTRY.list_keys()
    model_choices = DEFAULT_MODEL_REGISTRY.list_keys()
    strategy_choices = DEFAULT_STRATEGY_REGISTRY.list_keys()
    parser = argparse.ArgumentParser(description="Federated Perso-XAI baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Load, preprocess, partition, and persist a supported dataset.",
    )
    _add_common_dataset_args(prepare_parser, dataset_choices)
    _add_shared_path_args(prepare_parser)
    prepare_parser.add_argument("--global-eval-size", type=float, default=0.2)
    prepare_parser.add_argument("--client-test-size", type=float, default=0.2)
    prepare_parser.add_argument("--min-client-samples", type=int, default=10)
    prepare_parser.add_argument("--max-retries", type=int, default=50)

    centralized_parser = subparsers.add_parser(
        "train-centralized",
        help="Train the centralized logistic-regression baseline.",
    )
    centralized_parser.add_argument("--dataset", required=True, choices=dataset_choices)
    centralized_parser.add_argument("--seed", type=int, default=42)
    _add_shared_path_args(centralized_parser)
    _add_model_args(centralized_parser, model_choices)
    centralized_parser.add_argument("--skip-pooled-client-test", action="store_true")

    train_parser = subparsers.add_parser(
        "train-federated",
        help="Train a federated model from persisted client partitions.",
    )
    _add_common_dataset_args(train_parser, dataset_choices)
    _add_shared_path_args(train_parser)
    train_parser.add_argument("--rounds", type=int, default=10)
    _add_model_args(train_parser, model_choices)
    train_parser.add_argument("--strategy", choices=strategy_choices, default="fedavg")
    train_parser.add_argument("--fit-fraction", type=float, default=1.0)
    train_parser.add_argument("--evaluate-fraction", type=float, default=1.0)
    train_parser.add_argument("--min-available-clients", type=int, default=2)
    train_parser.add_argument(
        "--simulation-backend",
        choices=FEDERATED_BACKEND_CHOICES,
        default="auto",
    )
    train_parser.add_argument(
        "--debug-fallback-on-error",
        action="store_true",
        help="If Ray simulation fails, explicitly continue with the debug sequential runtime.",
    )
    train_parser.add_argument(
        "--secure-aggregation",
        action="store_true",
        help="Enable in-process simulated secure aggregation for shared parameters.",
    )
    train_parser.add_argument("--secure-num-helpers", type=int, default=5)
    train_parser.add_argument("--secure-privacy-threshold", type=int, default=2)
    train_parser.add_argument("--secure-reconstruction-threshold", type=int)
    train_parser.add_argument("--secure-field-modulus", type=int, default=2_147_483_647)
    train_parser.add_argument("--secure-quantization-scale", type=int, default=1 << 16)
    train_parser.add_argument("--secure-seed", type=int, default=0)
    train_parser.add_argument("--run-id")
    train_parser.add_argument(
        "--partitions",
        type=Path,
        help="Explicit path to the persisted partitioned client dataset root.",
    )
    train_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing completed federated run for the same deterministic output location.",
    )

    compare_parser = subparsers.add_parser(
        "compare-baselines",
        help="Compare centralized and federated predictive outputs.",
    )
    _add_common_dataset_args(compare_parser, dataset_choices)
    _add_shared_path_args(compare_parser)

    explain_parser = subparsers.add_parser(
        "explain-shap",
        help="Generate client-local SHAP explanations from a saved run artifact.",
    )
    _add_common_dataset_args(explain_parser, dataset_choices)
    _add_shared_path_args(explain_parser)
    explain_parser.add_argument("--client-id", type=int, required=True)
    explain_parser.add_argument("--model-source", choices=MODEL_SOURCE_CHOICES, default="federated")
    explain_parser.add_argument(
        "--partitions",
        type=Path,
        help="Explicit path to the persisted partitioned client dataset root.",
    )
    explain_parser.add_argument("--split", choices=["train", "test"], default="test")
    explain_parser.add_argument("--output", type=Path)
    explain_parser.add_argument("--background-sample-size", type=int)
    explain_parser.add_argument("--max-instances", type=int)
    explain_parser.add_argument(
        "--sampling-strategy",
        choices=["sequential", "random", "auto", "balanced", "quantile"],
    )
    explain_parser.add_argument("--random-state", type=int)
    explain_parser.add_argument("--shap-explainer-type", choices=["kernel", "sampling"])
    explain_parser.add_argument("--shap-nsamples", type=int)
    explain_parser.add_argument("--shap-l1-reg")
    explain_parser.add_argument("--shap-l1-reg-k", type=int)

    explain_eval_parser = subparsers.add_parser(
        "run-explain-eval-job",
        help="Run one standalone post-training explain+evaluate job for a federated run_id.",
    )
    _add_shared_path_args(explain_eval_parser)
    explain_eval_parser.add_argument("--run-id", required=True)
    explain_eval_parser.add_argument("--client-id", required=True)
    explain_eval_parser.add_argument("--split", choices=["train", "test"], default="test")
    explain_eval_parser.add_argument("--shard-id", default="shard_000")
    explain_eval_parser.add_argument("--explainer", required=True)
    explain_eval_parser.add_argument("--config-id", required=True)
    explain_eval_parser.add_argument("--max-instances", type=int, default=50)
    explain_eval_parser.add_argument("--rows-per-shard", type=int, default=1024)
    explain_eval_parser.add_argument("--random-state", type=int, default=42)
    explain_eval_parser.add_argument("--force", action="store_true")

    explain_eval_plan_parser = subparsers.add_parser(
        "plan-explain-eval-jobs",
        help="Plan a client x shard x explainer x config explain+evaluate matrix as JSONL.",
    )
    _add_shared_path_args(explain_eval_plan_parser)
    explain_eval_plan_parser.add_argument("--run-id", required=True)
    explain_eval_plan_parser.add_argument(
        "--clients",
        default="all",
        help="Comma-separated client ids, or 'all'. Accepts values like 0,1 or client_000,client_001.",
    )
    explain_eval_plan_parser.add_argument("--split", choices=["train", "test"], default="test")
    explain_eval_plan_parser.add_argument(
        "--explainers",
        default="all",
        help="Comma-separated explainer names, or 'all'.",
    )
    explain_eval_plan_parser.add_argument(
        "--configs",
        dest="config_ids",
        default="all",
        help="Comma-separated concrete config ids, or 'all'. Only list-valued matrix grids are expanded.",
    )
    explain_eval_plan_parser.add_argument("--max-instances", type=int, default=50)
    explain_eval_plan_parser.add_argument("--rows-per-shard", type=int, default=1024)
    explain_eval_plan_parser.add_argument("--random-state", type=int, default=42)
    explain_eval_plan_parser.add_argument("--output", type=Path, required=True)
    explain_eval_plan_parser.add_argument("--skip-existing", action="store_true")
    explain_eval_plan_parser.add_argument("--force", action="store_true")

    explain_eval_plan_item_parser = subparsers.add_parser(
        "run-explain-eval-plan-item",
        help="Run one JSONL explain+evaluate plan row, usually from a Slurm array task.",
    )
    _add_shared_path_args(explain_eval_plan_item_parser, default_to_none=True)
    explain_eval_plan_item_parser.add_argument("--plan", type=Path, required=True)
    explain_eval_plan_item_parser.add_argument(
        "--index",
        type=int,
        help="Plan row index. Defaults to SLURM_ARRAY_TASK_ID when omitted.",
    )
    explain_eval_plan_item_parser.add_argument("--force", action="store_true")

    aggregate_explain_eval_parser = subparsers.add_parser(
        "aggregate-explain-eval",
        help="Aggregate completed explain/evaluate shard outputs for one run/config.",
    )
    _add_shared_path_args(aggregate_explain_eval_parser)
    aggregate_explain_eval_parser.add_argument("--run-id", required=True)
    aggregate_explain_eval_parser.add_argument("--selection", dest="selection_id", required=True)
    aggregate_explain_eval_parser.add_argument("--explainer", dest="explainer_name", required=True)
    aggregate_explain_eval_parser.add_argument("--config", dest="config_id", required=True)
    aggregate_explain_eval_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Aggregate completed shards even if some discovered shard artifacts are incomplete.",
    )

    recommender_context_parser = subparsers.add_parser(
        "prepare-recommender-context",
        help="Build client-local context features for recommender labeling/training.",
    )
    _add_shared_path_args(recommender_context_parser)
    recommender_context_parser.add_argument("--run-id", required=True)
    recommender_context_parser.add_argument("--selection", dest="selection_id", required=True)
    recommender_context_parser.add_argument(
        "--explainers",
        default="all",
        help="Comma-separated explainer names, or 'all'.",
    )
    recommender_context_parser.add_argument(
        "--configs",
        dest="config_ids",
        default="all",
        help="Comma-separated config ids, or 'all'.",
    )
    recommender_context_parser.add_argument(
        "--clients",
        default="all",
        help="Comma-separated client ids like client_000,client_001, or 'all'.",
    )

    recommender_label_parser = subparsers.add_parser(
        "label-recommender-context",
        help="Simulate per-client users and label candidate contexts with pairwise preferences.",
    )
    _add_shared_path_args(recommender_label_parser)
    recommender_label_parser.add_argument("--run-id", required=True)
    recommender_label_parser.add_argument("--selection", dest="selection_id", required=True)
    recommender_label_parser.add_argument("--persona", default="lay")
    recommender_label_parser.add_argument("--persona-config", type=Path)
    recommender_label_parser.add_argument("--simulator", default="dirichlet_persona")
    recommender_label_parser.add_argument(
        "--clients",
        default="all",
        help="Comma-separated client ids like client_000,client_001, or 'all'.",
    )
    recommender_label_parser.add_argument(
        "--context-filename",
        default="candidate_context.parquet",
        help="Client-local context file to label, usually candidate_context.parquet or all_candidate_context.parquet.",
    )
    recommender_label_parser.add_argument("--label-filename", default="pairwise_labels.parquet")
    recommender_label_parser.add_argument("--seed", type=int, default=42)
    recommender_label_parser.add_argument("--label-seed", type=int, default=1729)
    recommender_label_parser.add_argument("--instance-test-size", type=float, default=0.2)
    recommender_label_parser.add_argument("--instance-split-seed", type=int)
    recommender_label_parser.add_argument("--tau", type=float)
    recommender_label_parser.add_argument("--concentration-c", type=float)

    recommender_train_parser = subparsers.add_parser(
        "train-recommender-federated",
        help="Train a federated pairwise explanation recommender from client-local labels.",
    )
    _add_shared_path_args(recommender_train_parser)
    recommender_train_parser.add_argument("--run-id", required=True)
    recommender_train_parser.add_argument("--selection", dest="selection_id", required=True)
    recommender_train_parser.add_argument("--persona", required=True)
    recommender_train_parser.add_argument("--clients", default="all")
    recommender_train_parser.add_argument("--context-filename", default="candidate_context.parquet")
    recommender_train_parser.add_argument("--label-filename", default="pairwise_labels.parquet")
    recommender_train_parser.add_argument(
        "--recommender",
        dest="recommender_type",
        choices=SUPPORTED_RECOMMENDER_TYPES,
        default=DEFAULT_RECOMMENDER_TYPE,
    )
    recommender_train_parser.add_argument("--rounds", type=int, default=10)
    recommender_train_parser.add_argument("--epochs", type=int, default=5)
    recommender_train_parser.add_argument("--batch-size", type=int, default=64)
    recommender_train_parser.add_argument("--learning-rate", type=float, default=0.05)
    recommender_train_parser.add_argument("--l2-regularization", type=float, default=0.0)
    recommender_train_parser.add_argument("--seed", type=int, default=42)
    recommender_train_parser.add_argument("--strategy", choices=strategy_choices, default="fedavg")
    recommender_train_parser.add_argument("--fit-fraction", type=float, default=1.0)
    recommender_train_parser.add_argument("--evaluate-fraction", type=float, default=1.0)
    recommender_train_parser.add_argument("--min-available-clients", type=int, default=2)
    recommender_train_parser.add_argument(
        "--simulation-backend",
        choices=FEDERATED_BACKEND_CHOICES,
        default="auto",
    )
    recommender_train_parser.add_argument("--debug-fallback-on-error", action="store_true")
    recommender_train_parser.add_argument(
        "--secure-aggregation",
        action="store_true",
        help="Enable in-process simulated secure aggregation for recommender parameters.",
    )
    recommender_train_parser.add_argument("--secure-num-helpers", type=int, default=5)
    recommender_train_parser.add_argument("--secure-privacy-threshold", type=int, default=2)
    recommender_train_parser.add_argument("--secure-reconstruction-threshold", type=int)
    recommender_train_parser.add_argument("--secure-field-modulus", type=int, default=2_147_483_647)
    recommender_train_parser.add_argument("--secure-quantization-scale", type=int, default=1 << 16)
    recommender_train_parser.add_argument("--secure-seed", type=int, default=0)
    recommender_train_parser.add_argument(
        "--clustered",
        action="store_true",
        help="Enable clustered recommender training with secure K-means and secure per-cluster aggregation.",
    )
    recommender_train_parser.add_argument(
        "--clustering-method",
        default="secure_kmeans",
        choices=["secure_kmeans"],
    )
    recommender_train_parser.add_argument("--clustering-k", type=int, default=3)
    recommender_train_parser.add_argument("--clustering-pca-components", type=int, default=8)
    recommender_train_parser.add_argument("--top-k", default="1,3,5")
    recommender_train_parser.add_argument("--force", action="store_true")

    recommender_eval_parser = subparsers.add_parser(
        "evaluate-recommender",
        help="Evaluate a trained global recommender against client-local labels.",
    )
    _add_shared_path_args(recommender_eval_parser)
    recommender_eval_parser.add_argument("--run-id", required=True)
    recommender_eval_parser.add_argument("--selection", dest="selection_id", required=True)
    recommender_eval_parser.add_argument("--persona", required=True)
    recommender_eval_parser.add_argument("--clients", default="all")
    recommender_eval_parser.add_argument("--context-filename", default="candidate_context.parquet")
    recommender_eval_parser.add_argument("--label-filename", default="pairwise_labels.parquet")
    recommender_eval_parser.add_argument(
        "--recommender",
        dest="recommender_type",
        choices=SUPPORTED_RECOMMENDER_TYPES,
        default=DEFAULT_RECOMMENDER_TYPE,
    )
    recommender_eval_parser.add_argument("--model-path", type=Path)
    recommender_eval_group = recommender_eval_parser.add_mutually_exclusive_group()
    recommender_eval_group.add_argument(
        "--secure-aggregation",
        dest="secure_aggregation",
        action="store_true",
        help="Load the secure recommender artifact path when both secure and plain outputs exist.",
    )
    recommender_eval_group.add_argument(
        "--plain-aggregation",
        dest="secure_aggregation",
        action="store_false",
        help="Load the plain recommender artifact path when both secure and plain outputs exist.",
    )
    recommender_eval_parser.set_defaults(secure_aggregation=None)
    recommender_eval_parser.add_argument("--top-k", default="1,3,5")
    recommender_eval_parser.add_argument("--output", type=Path)


    launcher_parser = subparsers.add_parser(
        "launch-experiment-jobs",
        help="Run a YAML-defined prepare/train matrix and create explain/evaluate Slurm array plans.",
    )
    launcher_parser.add_argument("--config", type=Path, required=True)
    launcher_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Expand and report launcher experiments without running preparation or training.",
    )
    launcher_parser.add_argument(
        "--submit-slurm",
        action="store_true",
        help="Submit generated explain/evaluate Slurm array scripts with sbatch.",
    )
    launcher_parser.add_argument(
        "--force",
        action="store_true",
        default=None,
        help="Overwrite existing completed federated training runs for launcher experiments.",
    )
    return parser


def main() -> None:
    """Run the CLI."""

    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-data":
        config = DataPreparationConfig(
            dataset_name=args.dataset,
            seed=args.seed,
            paths=_build_artifact_paths(args),
            preprocessing=PreprocessingConfig(
                global_eval_size=args.global_eval_size,
                client_test_size=args.client_test_size,
            ),
            partition=PartitionConfig(
                num_clients=args.num_clients,
                alpha=args.alpha,
                min_client_samples=args.min_client_samples,
                max_retries=args.max_retries,
            ),
        )
        result = prepare_federated_dataset(config)
        print(
            json.dumps(
                {
                    "prepared_root": str(result.prepared_artifacts.root_dir),
                    "partition_root": str(result.federated_artifacts.root_dir),
                    "feature_metadata_path": str(result.prepared_artifacts.feature_metadata_path),
                },
                indent=2,
            )
        )
        return

    if args.command == "train-centralized":
        config = CentralizedTrainingConfig(
            dataset_name=args.dataset,
            seed=args.seed,
            model_name=args.model,
            paths=_build_artifact_paths(args),
            model=_build_model_config(args),
            evaluate_on_pooled_client_test=not args.skip_pooled_client_test,
        )
        result_dir, summary = train_centralized_from_prepared(config)
        print(
            json.dumps(
                {
                    "result_dir": str(result_dir),
                    "global_eval_metrics": summary["evaluation"]["predictive"]["splits"]["global_eval"]["metrics"],
                    "pooled_client_test_metrics": (
                        summary["evaluation"]["predictive"]["splits"]
                        .get("pooled_client_test", {})
                        .get("metrics")
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "train-federated":
        config = FederatedTrainingConfig(
            dataset_name=args.dataset,
            seed=args.seed,
            model_name=args.model,
            paths=_build_artifact_paths(args),
            model=_build_model_config(args),
            num_clients=args.num_clients,
            alpha=args.alpha,
            strategy_name=args.strategy,
            rounds=args.rounds,
            fit_fraction=args.fit_fraction,
            evaluate_fraction=args.evaluate_fraction,
            min_available_clients=args.min_available_clients,
            simulation_backend=args.simulation_backend,
            debug_fallback_on_error=args.debug_fallback_on_error,
            secure_aggregation=args.secure_aggregation,
            secure_num_helpers=args.secure_num_helpers,
            secure_privacy_threshold=args.secure_privacy_threshold,
            secure_reconstruction_threshold=args.secure_reconstruction_threshold,
            secure_field_modulus=args.secure_field_modulus,
            secure_quantization_scale=args.secure_quantization_scale,
            secure_seed=args.secure_seed,
        )
        artifacts, summary = train_federated_from_partitions(
            config,
            run_id=args.run_id,
            partition_data_root=args.partitions,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(artifacts.run_dir),
                    "model_path": str(artifacts.model_artifact_path),
                    "training_metadata_path": str(artifacts.training_metadata_path),
                    "status": summary["status"],
                    "rounds_completed": summary["rounds_completed"],
                },
                indent=2,
            )
        )
        return

    if args.command == "compare-baselines":
        config = ComparisonConfig(
            dataset_name=args.dataset,
            seed=args.seed,
            num_clients=args.num_clients,
            alpha=args.alpha,
            paths=_build_artifact_paths(args),
        )
        report_path, report = compare_centralized_and_federated(config)
        print(
            json.dumps(
                {
                    "report_path": str(report_path),
                    "predictive_metric_comparison": report["predictive_metric_comparison"],
                },
                indent=2,
            )
        )
        return

    if args.command == "explain-shap":
        paths = _build_artifact_paths(args)
        partition_data_root = resolve_partition_data_root_for_explanations(
            paths=paths,
            dataset_name=args.dataset,
            seed=args.seed,
            model_source=args.model_source,
            num_clients=args.num_clients,
            alpha=args.alpha,
            partition_data_root=args.partitions,
        )
        client_data = load_client_data_for_explanations(
            paths=paths,
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            alpha=args.alpha,
            seed=args.seed,
            client_id=args.client_id,
            partition_data_root=partition_data_root,
        )
        model, result_dir = load_saved_model_for_explanations(
            paths=paths,
            dataset_name=args.dataset,
            seed=args.seed,
            model_source=args.model_source,
            num_clients=args.num_clients,
            alpha=args.alpha,
        )
        feature_names, feature_metadata_path = resolve_feature_names_for_explanations(
            paths=paths,
            dataset_name=args.dataset,
            seed=args.seed,
            model_source=args.model_source,
            num_clients=args.num_clients,
            alpha=args.alpha,
            partition_data_root=partition_data_root,
        )
        payload = generate_client_local_explanations(
            client_data=client_data,
            model=model,
            feature_names=feature_names,
            explainer_name="shap",
            split_name=args.split,
            params_override=_build_shap_override_args(args),
        )
        output_path = args.output or _default_explanation_output_path(
            result_dir=result_dir,
            client_id=args.client_id,
            split_name=args.split,
        )
        save_client_explanations(output_path, payload)
        print(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "model_source": args.model_source,
                    "partition_data_root": str(partition_data_root),
                    "client_id": args.client_id,
                    "split": args.split,
                    "n_explanations": payload["n_explanations"],
                    "feature_metadata_path": str(feature_metadata_path),
                },
                indent=2,
            )
        )
        return

    if args.command == "run-explain-eval-job":
        payload = run_explain_eval_job(
            run_id=args.run_id,
            client_id=args.client_id,
            split=args.split,
            shard_id=args.shard_id,
            explainer_name=args.explainer,
            config_id=args.config_id,
            max_instances=args.max_instances,
            rows_per_shard=args.rows_per_shard,
            random_state=args.random_state,
            force=args.force,
            paths=_build_artifact_paths(args),
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "plan-explain-eval-jobs":
        payload = plan_explain_eval_jobs(
            run_id=args.run_id,
            output_path=args.output,
            clients=args.clients,
            split=args.split,
            explainers=args.explainers,
            config_ids=args.config_ids,
            max_instances=args.max_instances,
            rows_per_shard=args.rows_per_shard,
            random_state=args.random_state,
            skip_existing=args.skip_existing,
            force=args.force,
            paths=_build_artifact_paths(args),
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "run-explain-eval-plan-item":
        index = args.index
        if index is None:
            try:
                index = int(os.environ["SLURM_ARRAY_TASK_ID"])
            except KeyError as exc:
                raise ValueError(
                    "--index is required when SLURM_ARRAY_TASK_ID is not set."
                ) from exc
        payload = run_explain_eval_plan_item(
            plan_path=args.plan,
            index=index,
            force=args.force,
            paths=_build_artifact_paths_or_none(args),
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "aggregate-explain-eval":
        payload = aggregate_explain_eval_results(
            run_id=args.run_id,
            selection_id=args.selection_id,
            explainer_name=args.explainer_name,
            config_id=args.config_id,
            allow_partial=args.allow_partial,
            paths=_build_artifact_paths(args),
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "prepare-recommender-context":
        payload = prepare_recommender_context(
            run_id=args.run_id,
            selection_id=args.selection_id,
            explainers=args.explainers,
            config_ids=args.config_ids,
            clients=args.clients,
            paths=_build_artifact_paths(args),
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "label-recommender-context":
        payload = label_recommender_context(
            run_id=args.run_id,
            selection_id=args.selection_id,
            persona=args.persona,
            persona_config_path=args.persona_config,
            simulator=args.simulator,
            clients=args.clients,
            context_filename=args.context_filename,
            label_filename=args.label_filename,
            seed=args.seed,
            label_seed=args.label_seed,
            instance_test_size=args.instance_test_size,
            instance_split_seed=args.instance_split_seed,
            tau=args.tau,
            concentration_c=args.concentration_c,
            paths=_build_artifact_paths(args),
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "train-recommender-federated":
        artifacts, metadata = train_federated_recommender(
            RecommenderFederatedTrainingConfig(
                run_id=args.run_id,
                selection_id=args.selection_id,
                persona=args.persona,
                recommender_type=args.recommender_type,
                paths=_build_artifact_paths(args),
                rounds=args.rounds,
                strategy_name=args.strategy,
                fit_fraction=args.fit_fraction,
                evaluate_fraction=args.evaluate_fraction,
                min_available_clients=args.min_available_clients,
                simulation_backend=args.simulation_backend,
                debug_fallback_on_error=args.debug_fallback_on_error,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                l2_regularization=args.l2_regularization,
                seed=args.seed,
                top_k=_parse_top_k(args.top_k),
                context_filename=args.context_filename,
                label_filename=args.label_filename,
                clients=args.clients,
                secure_aggregation=args.secure_aggregation,
                secure_num_helpers=args.secure_num_helpers,
                secure_privacy_threshold=args.secure_privacy_threshold,
                secure_reconstruction_threshold=args.secure_reconstruction_threshold,
                secure_field_modulus=args.secure_field_modulus,
                secure_quantization_scale=args.secure_quantization_scale,
                secure_seed=args.secure_seed,
                clustering=RecommenderClusteringConfig(
                    enabled=bool(args.clustered),
                    method=args.clustering_method,
                    k=args.clustering_k,
                    pca_components=args.clustering_pca_components,
                ),
            ),
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(artifacts.run_dir),
                    "model_artifact_path": str(artifacts.model_artifact_path),
                    "training_metadata_path": str(artifacts.training_metadata_path),
                    "evaluation_summary_path": str(artifacts.evaluation_summary_path),
                    "metadata": metadata,
                },
                indent=2,
            )
        )
        return

    if args.command == "evaluate-recommender":
        payload = evaluate_recommender_model(
            run_id=args.run_id,
            selection_id=args.selection_id,
            persona=args.persona,
            recommender_type=args.recommender_type,
            model_path=args.model_path,
            clients=args.clients,
            context_filename=args.context_filename,
            label_filename=args.label_filename,
            top_k=_parse_top_k(args.top_k),
            secure_aggregation=args.secure_aggregation,
            paths=_build_artifact_paths(args),
            output_path=args.output,
        )
        print(json.dumps(payload, indent=2))
        return


    if args.command == "launch-experiment-jobs":
        payload = run_job_launcher(
            config_path=args.config,
            dry_run=args.dry_run,
            submit_slurm=True if args.submit_slurm else None,
            force_training=args.force,
        )
        print(json.dumps(payload, indent=2))
        return

    raise ValueError(f"Unhandled command '{args.command}'.")


def _parse_top_k(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("--top-k must contain at least one integer.")
    parsed = tuple(sorted({int(item) for item in items}))
    if any(item < 1 for item in parsed):
        raise ValueError("--top-k values must be >= 1.")
    return parsed


def _add_common_dataset_args(parser: argparse.ArgumentParser, dataset_choices: list[str]) -> None:
    parser.add_argument("--dataset", required=True, choices=dataset_choices)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)


def _add_shared_path_args(
    parser: argparse.ArgumentParser,
    *,
    default_to_none: bool = False,
) -> None:
    defaults = {
        "prepared_root": None if default_to_none else Path("prepared"),
        "partition_root": None if default_to_none else Path("datasets"),
        "centralized_root": None if default_to_none else Path("centralized"),
        "federated_root": None if default_to_none else Path("federated"),
        "comparison_root": None if default_to_none else Path("comparisons"),
        "cache_dir": None if default_to_none else Path("data/cache/openml"),
    }
    parser.add_argument("--prepared-root", type=Path, default=defaults["prepared_root"])
    parser.add_argument("--partition-root", type=Path, default=defaults["partition_root"])
    parser.add_argument("--centralized-root", type=Path, default=defaults["centralized_root"])
    parser.add_argument("--federated-root", type=Path, default=defaults["federated_root"])
    parser.add_argument("--comparison-root", type=Path, default=defaults["comparison_root"])
    parser.add_argument("--cache-dir", type=Path, default=defaults["cache_dir"])


def _add_model_args(parser: argparse.ArgumentParser, model_choices: list[str]) -> None:
    parser.add_argument("--model", choices=model_choices, default="logistic_regression")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.0)


def _build_artifact_paths(args: argparse.Namespace) -> ArtifactPaths:
    return ArtifactPaths(
        prepared_root=args.prepared_root,
        partition_root=args.partition_root,
        centralized_root=args.centralized_root,
        federated_root=args.federated_root,
        comparison_root=args.comparison_root,
        cache_dir=args.cache_dir,
    )


def _build_artifact_paths_or_none(args: argparse.Namespace) -> ArtifactPaths | None:
    path_values = (
        args.prepared_root,
        args.partition_root,
        args.centralized_root,
        args.federated_root,
        args.comparison_root,
        args.cache_dir,
    )
    if all(value is None for value in path_values):
        return None
    if any(value is None for value in path_values):
        raise ValueError(
            "Either provide all artifact root overrides or none when running a plan item."
        )
    return _build_artifact_paths(args)


def _build_model_config(args: argparse.Namespace) -> LogisticRegressionConfig:
    return LogisticRegressionConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
    )


def _build_shap_override_args(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {"background_data_source": "client_local_train"}
    if args.background_sample_size is not None:
        overrides["background_sample_size"] = args.background_sample_size
    if args.max_instances is not None:
        overrides["max_instances"] = args.max_instances
    if args.sampling_strategy is not None:
        overrides["sampling_strategy"] = args.sampling_strategy
    if args.random_state is not None:
        overrides["random_state"] = args.random_state
    if args.shap_explainer_type is not None:
        overrides["shap_explainer_type"] = args.shap_explainer_type
    if args.shap_nsamples is not None:
        overrides["shap_nsamples"] = args.shap_nsamples
    if args.shap_l1_reg is not None:
        overrides["shap_l1_reg"] = args.shap_l1_reg
    if args.shap_l1_reg_k is not None:
        overrides["shap_l1_reg_k"] = args.shap_l1_reg_k
    return overrides


def _default_explanation_output_path(
    *,
    result_dir: Path,
    client_id: int,
    split_name: str,
) -> Path:
    return result_dir / "explanations" / f"client_{client_id}_shap_{split_name}.json"


if __name__ == "__main__":
    main()
