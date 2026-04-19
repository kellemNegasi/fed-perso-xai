"""Flower clients, strategy helpers, and simulation runtimes."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "ClientData",
    "DEFAULT_STRATEGY_REGISTRY",
    "FedAvgStrategyFactory",
    "FederatedLogisticRegressionClient",
    "FederatedRunRecorder",
    "FlowerRuntimePlan",
    "SimulationArtifacts",
    "StrategyRegistry",
    "StrategySpec",
    "TrackingFedAvg",
    "create_strategy_factory",
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
    "DEFAULT_STRATEGY_REGISTRY": ("fed_perso_xai.fl.strategy", "DEFAULT_STRATEGY_REGISTRY"),
    "FedAvgStrategyFactory": ("fed_perso_xai.fl.strategy", "FedAvgStrategyFactory"),
    "FederatedRunRecorder": ("fed_perso_xai.fl.strategy", "FederatedRunRecorder"),
    "StrategyRegistry": ("fed_perso_xai.fl.strategy", "StrategyRegistry"),
    "StrategySpec": ("fed_perso_xai.fl.strategy", "StrategySpec"),
    "TrackingFedAvg": ("fed_perso_xai.fl.strategy", "TrackingFedAvg"),
    "create_strategy_factory": ("fed_perso_xai.fl.strategy", "create_strategy_factory"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, attribute_name)
