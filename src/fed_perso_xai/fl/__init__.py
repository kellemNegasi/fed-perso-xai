"""Flower clients, strategy helpers, and simulation runtimes."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "ClientData",
    "FedAvgStrategyFactory",
    "FederatedLogisticRegressionClient",
    "FederatedRunRecorder",
    "FlowerRuntimePlan",
    "SimulationArtifacts",
    "TrackingFedAvg",
    "flower_support_available",
    "plan_flower_runtime",
    "require_flower_support",
    "run_federated_training",
]

_EXPORTS = {
    "ClientData": ("fed_perso_xai.fl.client", "ClientData"),
    "FederatedLogisticRegressionClient": (
        "fed_perso_xai.fl.client",
        "FederatedLogisticRegressionClient",
    ),
    "FlowerRuntimePlan": ("fed_perso_xai.fl.simulation", "FlowerRuntimePlan"),
    "SimulationArtifacts": ("fed_perso_xai.fl.simulation", "SimulationArtifacts"),
    "flower_support_available": ("fed_perso_xai.fl.simulation", "flower_support_available"),
    "plan_flower_runtime": ("fed_perso_xai.fl.simulation", "plan_flower_runtime"),
    "require_flower_support": ("fed_perso_xai.fl.simulation", "require_flower_support"),
    "run_federated_training": ("fed_perso_xai.fl.simulation", "run_federated_training"),
    "FedAvgStrategyFactory": ("fed_perso_xai.fl.strategy", "FedAvgStrategyFactory"),
    "FederatedRunRecorder": ("fed_perso_xai.fl.strategy", "FederatedRunRecorder"),
    "TrackingFedAvg": ("fed_perso_xai.fl.strategy", "TrackingFedAvg"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, attribute_name)
