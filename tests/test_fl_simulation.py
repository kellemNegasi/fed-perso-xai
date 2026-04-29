from __future__ import annotations

from fed_perso_xai.fl.simulation import _build_ray_backend_config, _slurm_cpu_limit


def test_build_ray_backend_config_uses_slurm_cpu_cap(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    monkeypatch.setattr("fed_perso_xai.fl.simulation.os.cpu_count", lambda: 256)

    backend_config = _build_ray_backend_config(
        simulation_resources={"num_cpus": 1.0},
        max_concurrent_clients=10,
    )

    assert backend_config["client_resources"] == {"num_cpus": 1.0}
    assert backend_config["init_args"]["ignore_reinit_error"] is True
    assert backend_config["init_args"]["num_cpus"] == 7.0


def test_build_ray_backend_config_caps_to_client_parallelism(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    monkeypatch.setattr("fed_perso_xai.fl.simulation.os.cpu_count", lambda: 256)

    backend_config = _build_ray_backend_config(
        simulation_resources={"num_cpus": 1.0},
        max_concurrent_clients=10,
    )

    assert backend_config["init_args"]["num_cpus"] == 10.0


def test_build_ray_backend_config_handles_fractional_client_cpu_budget(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    monkeypatch.setattr("fed_perso_xai.fl.simulation.os.cpu_count", lambda: 256)

    backend_config = _build_ray_backend_config(
        simulation_resources={"num_cpus": 0.5},
        max_concurrent_clients=10,
    )

    assert backend_config["init_args"]["num_cpus"] == 5.0


def test_slurm_cpu_limit_parses_slurm_job_cpu_list(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "72(x2),36")

    assert _slurm_cpu_limit() == 72.0
