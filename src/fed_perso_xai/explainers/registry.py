"""Explainer registry and YAML-backed config helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


class ExplainerRegistry:
    """Lightweight registry wrapper for `configs/explainers.yml`."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or (CONFIG_DIR / "explainers.yml")
        self._raw_config = self._load_yaml(self.config_path)
        self._entries = self._build_index()

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _build_index(self) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        for name, spec in self._raw_config.items():
            if not isinstance(spec, dict):
                continue
            if name.startswith("_") or name == "templates":
                continue
            normalized = dict(spec)
            supported = normalized.get("supported_data_types")
            normalized["supported_data_types"] = list(supported or ["tabular"])
            entries[name] = normalized
        return entries

    def get(self, name: str) -> dict[str, Any]:
        try:
            spec = self._entries[name]
        except KeyError as exc:
            supported = ", ".join(sorted(self._entries))
            raise KeyError(
                f"Unknown explainer entry '{name}' in {self.config_path}. Supported explainers: {supported}."
            ) from exc
        return copy.deepcopy(spec)

    def list_keys(self) -> list[str]:
        return sorted(self._entries)


def load_explainer_hyperparameter_grid(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load candidate-grid definitions from `explainer_hyperparameters.yml`."""

    config_path = path or (CONFIG_DIR / "explainer_hyperparameters.yml")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    explainers = payload.get("explainers")
    if not isinstance(explainers, dict):
        raise ValueError(f"Unexpected explainer hyperparameter format in {config_path}.")
    return copy.deepcopy(explainers)


DEFAULT_EXPLAINER_REGISTRY = ExplainerRegistry()
