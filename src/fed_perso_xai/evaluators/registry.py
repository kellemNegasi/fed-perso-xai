"""Metric registry and YAML-backed config helpers."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


class MetricRegistry:
    """Lightweight registry wrapper for `configs/metrics.yml`."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or (CONFIG_DIR / "metrics.yml")
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
            entries[name] = dict(spec)
        return entries

    def get(self, name: str) -> dict[str, Any]:
        try:
            spec = self._entries[name]
        except KeyError as exc:
            supported = ", ".join(sorted(self._entries))
            raise KeyError(
                f"Unknown metric entry '{name}' in {self.config_path}. Supported metrics: {supported}."
            ) from exc
        return copy.deepcopy(spec)

    def list_keys(self) -> list[str]:
        return sorted(self._entries)

    def is_available(self, name: str) -> bool:
        """Return whether the configured metric class is currently importable."""
        spec = self.get(name)
        module_name = str(spec["module"])
        class_name = str(spec["class"])
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            return True
        except Exception:
            return False


def load_metric_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load metric definitions from `metrics.yml`."""
    registry = MetricRegistry(config_path=path)
    return {name: registry.get(name) for name in registry.list_keys()}


DEFAULT_METRIC_REGISTRY = MetricRegistry()
