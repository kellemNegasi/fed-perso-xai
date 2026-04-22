"""CLI entrypoints for the stage-1 experiment workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fed_perso_xai.data.catalog import DEFAULT_DATASET_REGISTRY
from fed_perso_xai.fl.strategy import DEFAULT_STRATEGY_REGISTRY
from fed_perso_xai.models.registry import DEFAULT_MODEL_REGISTRY
from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.orchestration.explanations import (
    generate_client_local_explanations,
    load_client_data_for_explanations,
    load_saved_model_for_explanations,
    resolve_feature_names_for_explanations,
    save_client_explanations,
)
from fed_perso_xai.orchestration.stage_b_training import train_federated_stage_b
from fed_perso_xai.orchestration.training import (
    compare_centralized_and_federated,
    train_centralized_from_prepared,
)
from fed_perso_xai.utils.config import (
    ArtifactPaths,
    CentralizedTrainingConfig,
    ComparisonConfig,
    DataPreparationConfig,
    FederatedTrainingConfig,
    LogisticRegressionConfig,
    PartitionConfig,
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
    parser = argparse.ArgumentParser(description="Federated Perso-XAI stage-1 baseline.")
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
        help="Run standalone Stage B federated training from persisted client partitions.",
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
        help="Overwrite an existing completed Stage B run for the same deterministic output location.",
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
        artifacts, summary = train_federated_stage_b(
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
        client_data = load_client_data_for_explanations(
            paths=paths,
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            alpha=args.alpha,
            seed=args.seed,
            client_id=args.client_id,
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
                    "client_id": args.client_id,
                    "split": args.split,
                    "n_explanations": payload["n_explanations"],
                    "feature_metadata_path": str(feature_metadata_path),
                },
                indent=2,
            )
        )
        return

    raise ValueError(f"Unhandled command '{args.command}'.")


def _add_common_dataset_args(parser: argparse.ArgumentParser, dataset_choices: list[str]) -> None:
    parser.add_argument("--dataset", required=True, choices=dataset_choices)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)


def _add_shared_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared-root", type=Path, default=Path("prepared"))
    parser.add_argument("--partition-root", type=Path, default=Path("datasets"))
    parser.add_argument("--centralized-root", type=Path, default=Path("centralized"))
    parser.add_argument("--federated-root", type=Path, default=Path("federated"))
    parser.add_argument("--comparison-root", type=Path, default=Path("comparisons"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/openml"))


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
