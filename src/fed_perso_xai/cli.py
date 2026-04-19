"""CLI entrypoints for stage-1 dataset preparation and training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.orchestration.training import train_from_saved_partitions
from fed_perso_xai.utils.config import (
    DataPreparationConfig,
    FederatedTrainingConfig,
    PartitionConfig,
    PreprocessingConfig,
)
from fed_perso_xai.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(description="Federated Perso-XAI stage-1 baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Load, preprocess, partition, and persist a supported dataset.",
    )
    _add_common_dataset_args(prepare_parser)
    prepare_parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    prepare_parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/openml"))
    prepare_parser.add_argument("--global-eval-size", type=float, default=0.2)
    prepare_parser.add_argument("--client-test-size", type=float, default=0.2)
    prepare_parser.add_argument("--min-client-samples", type=int, default=10)
    prepare_parser.add_argument("--max-retries", type=int, default=50)

    train_parser = subparsers.add_parser(
        "train-federated",
        help="Train a Flower-based federated logistic regression model.",
    )
    _add_common_dataset_args(train_parser)
    train_parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    train_parser.add_argument("--results-root", type=Path, default=Path("results"))
    train_parser.add_argument("--rounds", type=int, default=10)
    train_parser.add_argument("--local-epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=0.05)
    train_parser.add_argument("--l2-regularization", type=float, default=0.0)
    train_parser.add_argument("--fit-fraction", type=float, default=1.0)
    train_parser.add_argument("--evaluate-fraction", type=float, default=1.0)
    train_parser.add_argument("--min-available-clients", type=int, default=2)
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
            output_root=args.output_root,
            cache_dir=args.cache_dir,
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
                    "partition_root": str(result.artifacts.root_dir),
                    "metadata_path": str(result.artifacts.metadata_path),
                    "feature_count": len(result.feature_names),
                },
                indent=2,
            )
        )
        return

    if args.command == "train-federated":
        config = FederatedTrainingConfig(
            dataset_name=args.dataset,
            seed=args.seed,
            data_root=args.data_root,
            results_root=args.results_root,
            num_clients=args.num_clients,
            alpha=args.alpha,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
            fit_fraction=args.fit_fraction,
            evaluate_fraction=args.evaluate_fraction,
            min_available_clients=args.min_available_clients,
        )
        artifacts, summary = train_from_saved_partitions(config)
        print(
            json.dumps(
                {
                    "metrics_path": str(artifacts.metrics_path),
                    "model_path": str(artifacts.model_path),
                    "aggregated_weighted": summary["aggregated_weighted"],
                },
                indent=2,
            )
        )
        return

    raise ValueError(f"Unhandled command '{args.command}'.")


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=["adult_income", "bank_marketing"])
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)


if __name__ == "__main__":
    main()
