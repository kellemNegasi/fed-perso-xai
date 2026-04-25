"""YAML-driven launcher for prepare/train/explain-evaluate job matrices."""

from __future__ import annotations

import itertools
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.orchestration.explain_eval import plan_explain_eval_jobs
from fed_perso_xai.orchestration.federated_training import train_federated_from_partitions
from fed_perso_xai.utils.config import (
    ArtifactPaths,
    DataPreparationConfig,
    FederatedTrainingConfig,
    LogisticRegressionConfig,
    PartitionConfig,
    PreprocessingConfig,
)
from fed_perso_xai.utils.paths import partition_root


@dataclass(frozen=True)
class LauncherExperiment:
    """One expanded prepare/train/plan experiment."""

    dataset_name: str
    seed: int
    num_clients: int
    alpha: float
    model_label: str
    model_name: str
    model_config: LogisticRegressionConfig
    rounds: int
    strategy_name: str
    simulation_backend: str


def run_job_launcher(
    *,
    config_path: Path,
    dry_run: bool = False,
    submit_slurm: bool | None = None,
    force_training: bool | None = None,
) -> dict[str, Any]:
    """Run a YAML-defined prepare/train/explain-evaluate matrix."""

    raw_config = _load_launcher_yaml(config_path)
    paths = _build_paths(raw_config.get("paths") or {})
    experiments = _expand_experiments(raw_config)
    explain_cfg = raw_config.get("explain_eval") or {}
    slurm_cfg = explain_cfg.get("slurm") or {}
    should_submit = bool(slurm_cfg.get("submit", False)) if submit_slurm is None else submit_slurm
    should_force_training = (
        bool(_get_nested(raw_config, ["training", "force"], False))
        if force_training is None
        else bool(force_training)
    )

    runs: list[dict[str, Any]] = []
    prepared_partitions: dict[tuple[str, int, int, float], Path] = {}
    for experiment_index, experiment in enumerate(experiments):
        run_record: dict[str, Any] = {
            "experiment_index": experiment_index,
            "dataset_name": experiment.dataset_name,
            "seed": experiment.seed,
            "num_clients": experiment.num_clients,
            "alpha": experiment.alpha,
            "model_label": experiment.model_label,
            "model_name": experiment.model_name,
            "dry_run": dry_run,
        }
        if dry_run:
            run_record["status"] = "planned_only"
            runs.append(run_record)
            continue

        partition_key = (
            experiment.dataset_name,
            experiment.seed,
            experiment.num_clients,
            experiment.alpha,
        )
        if partition_key not in prepared_partitions:
            data_config = DataPreparationConfig(
                dataset_name=experiment.dataset_name,
                seed=experiment.seed,
                paths=paths,
                preprocessing=_build_preprocessing_config(raw_config.get("preprocessing") or {}),
                partition=PartitionConfig(
                    num_clients=experiment.num_clients,
                    alpha=experiment.alpha,
                    min_client_samples=int(
                        _get_nested(raw_config, ["partition", "min_client_samples"], 10)
                    ),
                    max_retries=int(_get_nested(raw_config, ["partition", "max_retries"], 50)),
                ),
            )
            prepared = prepare_federated_dataset(data_config)
            prepared_partitions[partition_key] = prepared.federated_artifacts.root_dir
        run_record["partition_root"] = str(prepared_partitions[partition_key])

        training_config = _build_training_config(
            raw_config=raw_config,
            paths=paths,
            experiment=experiment,
        )
        training_artifacts, training_summary = train_federated_from_partitions(
            training_config,
            run_id=_render_run_id(raw_config.get("run_id_template"), experiment),
            partition_data_root=partition_root(
                paths.partition_root,
                experiment.dataset_name,
                experiment.num_clients,
                experiment.alpha,
                experiment.seed,
            ),
            force=should_force_training,
        )
        run_id = str(training_summary["run_id"])
        run_record.update(
            {
                "training_status": training_summary["status"],
                "run_id": run_id,
                "training_run_dir": str(training_artifacts.run_dir),
            }
        )

        if bool(explain_cfg.get("enabled", True)):
            plan_path = _plan_path(
                explain_cfg=explain_cfg,
                experiment=experiment,
                run_id=run_id,
            )
            plan_summary = plan_explain_eval_jobs(
                run_id=run_id,
                output_path=plan_path,
                clients=str(explain_cfg.get("clients", "all")),
                split=str(explain_cfg.get("split", "test")),
                explainers=str(explain_cfg.get("explainers", "all")),
                config_ids=str(explain_cfg.get("configs", "all")),
                max_instances=int(explain_cfg.get("max_instances", 50)),
                random_state=int(explain_cfg.get("random_state", 42)),
                skip_existing=bool(explain_cfg.get("skip_existing", False)),
                force=bool(explain_cfg.get("force", False)),
                paths=paths,
            )
            run_record["explain_eval_plan"] = plan_summary
            if plan_summary["job_count"] > 0 and bool(slurm_cfg.get("enabled", True)):
                script_path = _write_slurm_array_script(
                    slurm_cfg=slurm_cfg,
                    plan_path=plan_path,
                    array_range=str(plan_summary["array_range"]),
                    run_id=run_id,
                )
                run_record["slurm_script_path"] = str(script_path)
                if should_submit:
                    run_record["slurm_submission"] = _submit_slurm_job(script_path, plan_path)
        runs.append(run_record)

    return {
        "status": "completed" if not dry_run else "dry_run",
        "config_path": str(config_path),
        "experiment_count": len(experiments),
        "runs": runs,
    }


def _load_launcher_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Launcher config must be a mapping: {config_path}")
    return payload


def _build_paths(raw_paths: dict[str, Any]) -> ArtifactPaths:
    return ArtifactPaths(
        prepared_root=Path(str(raw_paths.get("prepared_root", "prepared"))),
        partition_root=Path(str(raw_paths.get("partition_root", "datasets"))),
        centralized_root=Path(str(raw_paths.get("centralized_root", "centralized"))),
        federated_root=Path(str(raw_paths.get("federated_root", "federated"))),
        comparison_root=Path(str(raw_paths.get("comparison_root", "comparisons"))),
        cache_dir=Path(str(raw_paths.get("cache_dir", "data/cache/openml"))),
    )


def _expand_experiments(raw_config: dict[str, Any]) -> list[LauncherExperiment]:
    datasets = _as_list(raw_config.get("datasets") or raw_config.get("dataset"))
    if not datasets:
        raise ValueError("Launcher config must define at least one dataset.")
    seeds = [int(seed) for seed in _as_list(raw_config.get("seeds", [42]))]
    partition_cfg = raw_config.get("partition") or {}
    num_clients_values = [
        int(value)
        for value in _as_list(partition_cfg.get("num_clients") or partition_cfg.get("client_counts"))
    ]
    alpha_values = [
        float(value)
        for value in _as_list(partition_cfg.get("alphas") or partition_cfg.get("alpha"))
    ]
    if not num_clients_values:
        raise ValueError("Launcher config must define partition.num_clients or partition.client_counts.")
    if not alpha_values:
        raise ValueError("Launcher config must define partition.alphas or partition.alpha.")

    training_cfg = raw_config.get("training") or {}
    rounds_values = [int(value) for value in _as_list(training_cfg.get("rounds", [10]))]
    strategy_values = [str(value) for value in _as_list(training_cfg.get("strategy", "fedavg"))]
    backend_values = [str(value) for value in _as_list(training_cfg.get("simulation_backend", "auto"))]
    model_entries = _expand_model_entries(raw_config.get("models") or raw_config.get("model"))

    experiments: list[LauncherExperiment] = []
    for dataset_name, seed, num_clients, alpha, model_entry, rounds, strategy_name, backend in itertools.product(
        [str(dataset) for dataset in datasets],
        seeds,
        num_clients_values,
        alpha_values,
        model_entries,
        rounds_values,
        strategy_values,
        backend_values,
    ):
        experiments.append(
            LauncherExperiment(
                dataset_name=dataset_name,
                seed=seed,
                num_clients=num_clients,
                alpha=alpha,
                model_label=str(model_entry["label"]),
                model_name=str(model_entry["name"]),
                model_config=model_entry["config"],
                rounds=rounds,
                strategy_name=strategy_name,
                simulation_backend=backend,
            )
        )
    return experiments


def _expand_model_entries(raw_models: Any) -> list[dict[str, Any]]:
    if raw_models is None:
        raw_models = [{"name": "logistic_regression"}]
    if isinstance(raw_models, dict) and "name" not in raw_models:
        raw_models = [
            {"name": model_name, **(model_cfg or {})}
            for model_name, model_cfg in raw_models.items()
        ]

    entries: list[dict[str, Any]] = []
    for raw_model in _as_list(raw_models):
        if not isinstance(raw_model, dict):
            raw_model = {"name": str(raw_model)}
        model_name = str(raw_model.get("name", "logistic_regression"))
        params = raw_model.get("params") or {
            key: raw_model[key]
            for key in ("epochs", "batch_size", "learning_rate", "l2_regularization")
            if key in raw_model
        }
        param_grid = {
            "epochs": [int(value) for value in _as_list(params.get("epochs", 5))],
            "batch_size": [int(value) for value in _as_list(params.get("batch_size", 64))],
            "learning_rate": [float(value) for value in _as_list(params.get("learning_rate", 0.05))],
            "l2_regularization": [
                float(value)
                for value in _as_list(params.get("l2_regularization", 0.0))
            ],
        }
        for epochs, batch_size, learning_rate, l2_regularization in itertools.product(
            param_grid["epochs"],
            param_grid["batch_size"],
            param_grid["learning_rate"],
            param_grid["l2_regularization"],
        ):
            config = LogisticRegressionConfig(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
            )
            label = raw_model.get("label") or (
                f"{model_name}-epochs{epochs}-batch{batch_size}-lr{learning_rate}-l2{l2_regularization}"
            )
            entries.append({"label": label, "name": model_name, "config": config})
    return entries


def _build_preprocessing_config(raw_config: dict[str, Any]) -> PreprocessingConfig:
    return PreprocessingConfig(
        global_eval_size=float(raw_config.get("global_eval_size", 0.2)),
        client_test_size=float(raw_config.get("client_test_size", 0.2)),
        fitting_mode=str(raw_config.get("fitting_mode", "global_shared")),
        numeric_imputation_strategy=str(raw_config.get("numeric_imputation_strategy", "median")),
        categorical_imputation_strategy=str(
            raw_config.get("categorical_imputation_strategy", "most_frequent")
        ),
    )


def _build_training_config(
    *,
    raw_config: dict[str, Any],
    paths: ArtifactPaths,
    experiment: LauncherExperiment,
) -> FederatedTrainingConfig:
    training_cfg = raw_config.get("training") or {}
    secure_cfg = training_cfg.get("secure_aggregation") or {}
    secure_enabled = bool(secure_cfg) if isinstance(secure_cfg, dict) else bool(secure_cfg)
    return FederatedTrainingConfig(
        dataset_name=experiment.dataset_name,
        seed=experiment.seed,
        model_name=experiment.model_name,
        paths=paths,
        model=experiment.model_config,
        num_clients=experiment.num_clients,
        alpha=experiment.alpha,
        strategy_name=experiment.strategy_name,
        rounds=experiment.rounds,
        fit_fraction=float(training_cfg.get("fit_fraction", 1.0)),
        evaluate_fraction=float(training_cfg.get("evaluate_fraction", 1.0)),
        min_available_clients=int(training_cfg.get("min_available_clients", 2)),
        simulation_backend=experiment.simulation_backend,
        debug_fallback_on_error=bool(training_cfg.get("debug_fallback_on_error", False)),
        secure_aggregation=secure_enabled,
        secure_num_helpers=int(_dict_get(secure_cfg, "num_helpers", 5)),
        secure_privacy_threshold=int(_dict_get(secure_cfg, "privacy_threshold", 2)),
        secure_reconstruction_threshold=_optional_int(
            _dict_get(secure_cfg, "reconstruction_threshold", None)
        ),
        secure_field_modulus=int(_dict_get(secure_cfg, "field_modulus", 2_147_483_647)),
        secure_quantization_scale=int(_dict_get(secure_cfg, "quantization_scale", 1 << 16)),
        secure_seed=int(_dict_get(secure_cfg, "seed", 0)),
    )


def _plan_path(*, explain_cfg: dict[str, Any], experiment: LauncherExperiment, run_id: str) -> Path:
    plan_dir = Path(str(explain_cfg.get("plan_dir", "job_launcher/plans")))
    safe_run_id = _safe_segment(run_id)
    filename = (
        f"{experiment.dataset_name}__clients-{experiment.num_clients}"
        f"__alpha-{experiment.alpha}__seed-{experiment.seed}"
        f"__{experiment.model_label}__{safe_run_id}.jsonl"
    )
    return plan_dir / filename


def _write_slurm_array_script(
    *,
    slurm_cfg: dict[str, Any],
    plan_path: Path,
    array_range: str,
    run_id: str,
) -> Path:
    script_dir = Path(str(slurm_cfg.get("script_dir", "job_launcher/slurm")))
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"explain_eval__{_safe_segment(run_id)}.sbatch"
    concurrency = slurm_cfg.get("array_concurrency")
    array_spec = f"{array_range}%{int(concurrency)}" if concurrency else array_range
    job_name = str(slurm_cfg.get("job_name", "explain-eval"))
    log_dir = Path(str(slurm_cfg.get("log_dir", "logs")))
    log_dir.mkdir(parents=True, exist_ok=True)
    sbatch_args = _merged_sbatch_args(slurm_cfg)
    project_root = Path(str(slurm_cfg.get("project_root", Path.cwd()))).resolve()
    module_load = str(slurm_cfg.get("module_load", "") or "").strip()
    venv_path = str(slurm_cfg.get("venv_path", ".venv/bin/activate")).strip()
    sbatch_lines = [
        "#!/bin/bash",
        "#",
        "# SLURM array script generated by fed-perso-xai launch-experiment-jobs.",
        "# Runs one explain/evaluate JSONL plan row per array task.",
        "#",
        f"# Example: sbatch --array={array_spec} {script_path.name} {plan_path.resolve()}",
        "#",
        "# Job metadata -----------------------------------------------------------------",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array={array_spec}",
    ]
    for arg in sbatch_args:
        text = str(arg).strip()
        if text:
            sbatch_lines.append(f"#SBATCH {text}" if text.startswith("--") else f"#SBATCH {text}")
    sbatch_lines.extend(
        [
            "",
            "# Environment setup ------------------------------------------------------------",
            "set -euo pipefail",
            "",
            f'PROJECT_ROOT="${{PROJECT_ROOT:-{project_root}}}"',
            'cd "$PROJECT_ROOT"',
        ]
    )
    if module_load:
        sbatch_lines.append(f"module load {module_load}")
    if venv_path:
        sbatch_lines.extend(
            [
                f'if [[ -f "{venv_path}" ]]; then',
                f'  source "{venv_path}"',
                "fi",
            ]
        )
    sbatch_lines.extend(
        [
            "",
            'if ! command -v python3 >/dev/null 2>&1; then',
            '  echo "ERROR: python3 not found. Check module/venv setup." >&2',
            "  exit 1",
            "fi",
            "",
            "# Workload ---------------------------------------------------------------------",
            'PLAN="${1:-' + str(plan_path.resolve()) + '}"',
            'if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then',
            '  echo "ERROR: SLURM_ARRAY_TASK_ID is not set; submit with sbatch --array." >&2',
            "  exit 2",
            "fi",
            'if [[ ! -f "$PLAN" ]]; then',
            '  echo "ERROR: plan file does not exist: $PLAN" >&2',
            "  exit 2",
            "fi",
            'JOB_ID="${SLURM_ARRAY_TASK_ID}"',
            'NUM_JOBS="${SLURM_ARRAY_TASK_COUNT:-1}"',
            'PLAN_ROWS="$(wc -l < "$PLAN")"',
            'if (( JOB_ID < 0 || JOB_ID >= PLAN_ROWS )); then',
            '  echo "ERROR: SLURM_ARRAY_TASK_ID=${JOB_ID} out of range for plan rows=${PLAN_ROWS}." >&2',
            "  exit 2",
            "fi",
            "",
            'echo "[$(date --iso-8601=seconds)] Starting explain/evaluate array task"',
            'echo "  run_id=' + run_id + '"',
            'echo "  job_id=${JOB_ID}"',
            'echo "  num_jobs=${NUM_JOBS}"',
            'echo "  plan_rows=${PLAN_ROWS}"',
            'echo "  plan=${PLAN}"',
            'echo "  project_root=${PROJECT_ROOT}"',
            'echo "  host=${HOSTNAME}"',
            "",
            "python3 -m fed_perso_xai run-explain-eval-plan-item \\",
            '  --plan "$PLAN"',
            "",
            'echo "[$(date --iso-8601=seconds)] Task completed: ${JOB_ID}"',
            "",
        ]
    )
    script_path.write_text("\n".join(sbatch_lines), encoding="utf-8")
    return script_path


def _merged_sbatch_args(slurm_cfg: dict[str, Any]) -> list[str]:
    explicit_args = [str(arg).strip() for arg in _as_list(slurm_cfg.get("sbatch_args", [])) if str(arg).strip()]
    log_dir = str(slurm_cfg.get("log_dir", "logs")).rstrip("/") or "."
    defaults = {
        "output": f"{log_dir}/slurm-%x.%A_%a.out",
        "nodes": "1",
        "ntasks": "1",
    }
    merged: list[str] = []
    present = {_sbatch_option_name(arg) for arg in explicit_args}
    for key, value in defaults.items():
        if key not in present and f"{key}=" not in present:
            merged.append(f"--{key}={value}")
    merged.extend(explicit_args)
    return merged


def _sbatch_option_name(arg: str) -> str:
    text = arg.strip()
    if text.startswith("--"):
        text = text[2:]
    elif text.startswith("-"):
        text = text[1:]
    return text.split("=", 1)[0].split(None, 1)[0]


def _submit_slurm_job(script_path: Path, plan_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["sbatch", str(script_path), str(plan_path.resolve())],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _render_run_id(template: Any, experiment: LauncherExperiment) -> str | None:
    if template is None:
        return None
    return str(template).format(
        dataset=experiment.dataset_name,
        seed=experiment.seed,
        num_clients=experiment.num_clients,
        alpha=experiment.alpha,
        model_label=experiment.model_label,
        model_name=experiment.model_name,
        rounds=experiment.rounds,
        strategy=experiment.strategy_name,
        simulation_backend=experiment.simulation_backend,
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _dict_get(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _get_nested(payload: dict[str, Any], keys: list[str], default: Any) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _safe_segment(value: str) -> str:
    return (
        str(value)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "-")
        .replace(":", "-")
    )
