from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fed_perso_xai.cli import main
from fed_perso_xai.orchestration import job_launcher
from fed_perso_xai.orchestration.job_launcher import run_job_launcher


def _write_launcher_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    payload = {
        "paths": {
            "prepared_root": str(tmp_path / "prepared"),
            "partition_root": str(tmp_path / "datasets"),
            "federated_root": str(tmp_path / "federated"),
            "centralized_root": str(tmp_path / "centralized"),
            "comparison_root": str(tmp_path / "comparisons"),
            "cache_dir": str(tmp_path / "cache"),
        },
        "datasets": ["toy"],
        "seeds": [7],
        "partition": {
            "num_clients": [3],
            "alphas": [1.0],
            "min_client_samples": 2,
            "max_retries": 5,
        },
        "models": [
            {
                "label": "logreg-small",
                "name": "logistic_regression",
                "params": {
                    "epochs": 2,
                    "batch_size": 4,
                    "learning_rate": 0.1,
                    "l2_regularization": 0.0,
                },
            }
        ],
        "training": {
            "rounds": 1,
            "strategy": "fedavg",
            "simulation_backend": "debug-sequential",
            "force": False,
        },
        "explain_eval": {
            "enabled": True,
            "clients": "all",
            "split": "test",
            "explainers": "lime",
            "configs": "lime__kernel-1.5__samples-50",
            "max_instances": 5,
            "random_state": 11,
            "skip_existing": True,
            "plan_dir": str(tmp_path / "plans"),
            "slurm": {
                "enabled": True,
                "submit": False,
                "script_dir": str(tmp_path / "slurm"),
                "job_name": "xai-test",
                "array_concurrency": 2,
                "sbatch_args": ["--cpus-per-task=1", "--mem=1G"],
            },
        },
    }
    if overrides:
        payload.update(overrides)
    config_path = tmp_path / "launcher.yml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_job_launcher_dry_run_expands_yaml_matrix(tmp_path) -> None:
    config_path = _write_launcher_config(
        tmp_path,
        overrides={
            "datasets": ["toy_a", "toy_b"],
            "seeds": [1, 2],
            "partition": {"num_clients": [3, 5], "alphas": [0.5, 1.0]},
        },
    )

    summary = run_job_launcher(config_path=config_path, dry_run=True)

    assert summary["status"] == "dry_run"
    assert summary["experiment_count"] == 16
    assert len(summary["runs"]) == 16
    assert {run["dry_run"] for run in summary["runs"]} == {True}


def test_job_launcher_rejects_list_valued_training_sampling_fields(tmp_path) -> None:
    config_path = _write_launcher_config(
        tmp_path,
        overrides={
            "training": {
                "rounds": 1,
                "simulation_backend": "debug-sequential",
                "fit_fraction": [0.5, 1.0],
                "evaluate_fraction": [0.5, 1.0],
                "min_available_clients": [2, 3],
            },
        },
    )

    with pytest.raises(ValueError) as excinfo:
        run_job_launcher(config_path=config_path, dry_run=True)

    message = str(excinfo.value)
    assert "Unsupported list-valued launcher field(s)" in message
    assert "training.fit_fraction" in message
    assert "training.evaluate_fraction" in message
    assert "training.min_available_clients" in message
    assert "Use a single scalar value for now" in message


@pytest.mark.parametrize(
    ("overrides", "expected_field"),
    [
        ({"seeds": []}, "seeds"),
        ({"training": {"rounds": []}}, "training.rounds"),
        ({"training": {"strategy": []}}, "training.strategy"),
        ({"training": {"simulation_backend": []}}, "training.simulation_backend"),
        ({"models": []}, "models or model"),
        ({"models": [{"name": "logistic_regression", "params": {"epochs": []}}]}, "params.epochs"),
    ],
)
def test_job_launcher_rejects_empty_matrix_dimensions(
    tmp_path, overrides, expected_field
) -> None:
    config_path = _write_launcher_config(tmp_path, overrides=overrides)

    with pytest.raises(ValueError) as excinfo:
        run_job_launcher(config_path=config_path, dry_run=True)

    message = str(excinfo.value)
    assert "Launcher config must define at least one value" in message
    assert expected_field in message


def test_job_launcher_rejects_yaml_list_client_selector(tmp_path) -> None:
    config_path = _write_launcher_config(
        tmp_path,
        overrides={
            "explain_eval": {
                "enabled": True,
                "clients": [0, 1],
                "explainers": "lime",
                "configs": "lime__kernel-1.5__samples-50",
            },
        },
    )

    with pytest.raises(ValueError) as excinfo:
        run_job_launcher(config_path=config_path, dry_run=True)

    message = str(excinfo.value)
    assert "Unsupported list-valued launcher field(s)" in message
    assert "explain_eval.clients" in message
    assert "comma-separated string" in message


def test_job_launcher_accepts_yaml_list_explain_eval_explainers_and_configs(
    tmp_path, monkeypatch
) -> None:
    config_path = _write_launcher_config(
        tmp_path,
        overrides={
            "explain_eval": {
                "enabled": True,
                "clients": "all",
                "split": "test",
                "explainers": ["lime", "shap"],
                "configs": [
                    "lime__kernel-1.5__samples-50",
                    "shap__background-10__nsamples-50",
                ],
                "max_instances": 5,
                "random_state": 11,
                "skip_existing": True,
                "plan_dir": str(tmp_path / "plans"),
                "slurm": {"enabled": False},
            },
        },
    )
    calls: dict[str, object] = {}

    def fake_prepare(config):
        root = tmp_path / "datasets" / "toy" / "3_clients" / "alpha_1.0" / "seed_7"
        root.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(federated_artifacts=SimpleNamespace(root_dir=root))

    def fake_train(config, *, run_id=None, partition_data_root=None, force=False):
        return SimpleNamespace(run_dir=tmp_path / "federated_run"), {
            "status": "completed",
            "run_id": "run-xyz",
        }

    def fake_plan(**kwargs):
        calls["plan"] = kwargs
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("{}", encoding="utf-8")
        return {
            "status": "planned",
            "job_count": 1,
            "array_range": "0-0",
            "plan_path": str(kwargs["output_path"]),
        }

    monkeypatch.setattr(job_launcher, "prepare_federated_dataset", fake_prepare)
    monkeypatch.setattr(job_launcher, "train_federated_from_partitions", fake_train)
    monkeypatch.setattr(job_launcher, "plan_explain_eval_jobs", fake_plan)

    summary = run_job_launcher(config_path=config_path)

    assert summary["status"] == "completed"
    assert calls["plan"]["explainers"] == "lime,shap"
    assert (
        calls["plan"]["config_ids"]
        == "lime__kernel-1.5__samples-50,shap__background-10__nsamples-50"
    )


def test_job_launcher_runs_prepare_train_and_writes_slurm_script(tmp_path, monkeypatch) -> None:
    config_path = _write_launcher_config(tmp_path)
    calls: dict[str, object] = {}

    def fake_prepare(config):
        calls["prepare"] = config
        root = tmp_path / "datasets" / "toy" / "3_clients" / "alpha_1.0" / "seed_7"
        root.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            federated_artifacts=SimpleNamespace(root_dir=root),
        )

    def fake_train(config, *, run_id=None, partition_data_root=None, force=False):
        calls["train"] = {
            "config": config,
            "run_id": run_id,
            "partition_data_root": partition_data_root,
            "force": force,
        }
        return SimpleNamespace(run_dir=tmp_path / "federated_run"), {
            "status": "completed",
            "run_id": "run-xyz",
        }

    def fake_plan(**kwargs):
        calls["plan"] = kwargs
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("{}", encoding="utf-8")
        return {
            "status": "planned",
            "job_count": 3,
            "array_range": "0-2",
            "plan_path": str(kwargs["output_path"]),
        }

    monkeypatch.setattr(job_launcher, "prepare_federated_dataset", fake_prepare)
    monkeypatch.setattr(job_launcher, "train_federated_from_partitions", fake_train)
    monkeypatch.setattr(job_launcher, "plan_explain_eval_jobs", fake_plan)

    summary = run_job_launcher(config_path=config_path)

    run = summary["runs"][0]
    script_path = Path(run["slurm_script_path"])
    assert summary["status"] == "completed"
    assert calls["prepare"].dataset_name == "toy"
    assert calls["train"]["config"].model.epochs == 2
    assert calls["plan"]["run_id"] == "run-xyz"
    assert calls["plan"]["config_ids"] == "lime__kernel-1.5__samples-50"
    assert script_path.exists()
    script_text = script_path.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2%2" in script_text
    assert "#SBATCH --output=logs/slurm-%x.%A_%a.out" in script_text
    assert "#SBATCH --nodes=1" in script_text
    assert "#SBATCH --ntasks=1" in script_text
    assert "module load" not in script_text
    assert 'source ".venv/bin/activate"' in script_text
    assert "SLURM_ARRAY_TASK_ID is not set" in script_text
    assert "plan file does not exist" in script_text
    assert "PLAN_ROWS=" in script_text
    assert "out of range for plan rows" in script_text
    assert "Starting explain/evaluate array task" in script_text
    assert "run-explain-eval-plan-item" in script_text


def test_job_launcher_fails_when_sbatch_submission_fails(tmp_path, monkeypatch) -> None:
    config_path = _write_launcher_config(tmp_path)

    def fake_prepare(config):
        root = tmp_path / "datasets" / "toy" / "3_clients" / "alpha_1.0" / "seed_7"
        root.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(federated_artifacts=SimpleNamespace(root_dir=root))

    def fake_train(config, *, run_id=None, partition_data_root=None, force=False):
        return SimpleNamespace(run_dir=tmp_path / "federated_run"), {
            "status": "completed",
            "run_id": "run-xyz",
        }

    def fake_plan(**kwargs):
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text("{}", encoding="utf-8")
        return {
            "status": "planned",
            "job_count": 3,
            "array_range": "0-2",
            "plan_path": str(kwargs["output_path"]),
        }

    def fake_sbatch(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="sbatch: error: invalid account",
        )

    monkeypatch.setattr(job_launcher, "prepare_federated_dataset", fake_prepare)
    monkeypatch.setattr(job_launcher, "train_federated_from_partitions", fake_train)
    monkeypatch.setattr(job_launcher, "plan_explain_eval_jobs", fake_plan)
    monkeypatch.setattr(job_launcher.subprocess, "run", fake_sbatch)

    with pytest.raises(RuntimeError) as excinfo:
        run_job_launcher(config_path=config_path, submit_slurm=True)

    message = str(excinfo.value)
    assert "sbatch submission failed" in message
    assert "return code 1" in message
    assert "invalid account" in message


def test_job_launcher_prepares_each_partition_once_for_multiple_models(tmp_path, monkeypatch) -> None:
    config_path = _write_launcher_config(
        tmp_path,
        overrides={
            "models": [
                {"label": "model-a", "name": "logistic_regression", "params": {"epochs": 2}},
                {"label": "model-b", "name": "logistic_regression", "params": {"epochs": 3}},
            ],
            "explain_eval": {"enabled": False},
            "training": {"rounds": 1, "simulation_backend": "debug-sequential", "force": True},
        },
    )
    prepare_calls: list[object] = []

    def fake_prepare(config):
        prepare_calls.append(config)
        root = tmp_path / "datasets" / "toy" / "3_clients" / "alpha_1.0" / "seed_7"
        root.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(federated_artifacts=SimpleNamespace(root_dir=root))

    def fake_train(config, *, run_id=None, partition_data_root=None, force=False):
        return SimpleNamespace(run_dir=tmp_path / f"run-{config.model.epochs}"), {
            "status": "completed",
            "run_id": f"run-{config.model.epochs}",
        }

    monkeypatch.setattr(job_launcher, "prepare_federated_dataset", fake_prepare)
    monkeypatch.setattr(job_launcher, "train_federated_from_partitions", fake_train)

    summary = run_job_launcher(config_path=config_path)

    assert summary["experiment_count"] == 2
    assert len(prepare_calls) == 1


def test_job_launcher_force_training_override_takes_precedence(tmp_path, monkeypatch) -> None:
    config_path = _write_launcher_config(
        tmp_path,
        overrides={"explain_eval": {"enabled": False}},
    )
    calls: dict[str, object] = {}

    def fake_prepare(config):
        root = tmp_path / "datasets" / "toy" / "3_clients" / "alpha_1.0" / "seed_7"
        root.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(federated_artifacts=SimpleNamespace(root_dir=root))

    def fake_train(config, *, run_id=None, partition_data_root=None, force=False):
        calls["force"] = force
        return SimpleNamespace(run_dir=tmp_path / "federated_run"), {
            "status": "completed",
            "run_id": "run-xyz",
        }

    monkeypatch.setattr(job_launcher, "prepare_federated_dataset", fake_prepare)
    monkeypatch.setattr(job_launcher, "train_federated_from_partitions", fake_train)

    summary = run_job_launcher(config_path=config_path, force_training=True)

    assert summary["status"] == "completed"
    assert calls["force"] is True


def test_launch_experiment_jobs_cli_uses_launcher(tmp_path, monkeypatch, capsys) -> None:
    config_path = _write_launcher_config(tmp_path)
    calls: dict[str, object] = {}

    def fake_launcher(**kwargs):
        calls.update(kwargs)
        return {"status": "dry_run", "experiment_count": 1, "runs": []}

    monkeypatch.setattr("fed_perso_xai.cli.run_job_launcher", fake_launcher)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fed-perso-xai",
            "launch-experiment-jobs",
            "--config",
            str(config_path),
            "--dry-run",
            "--force",
        ],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "dry_run"
    assert calls["config_path"] == config_path
    assert calls["dry_run"] is True
    assert calls["force_training"] is True
