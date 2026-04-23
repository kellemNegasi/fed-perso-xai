"""Explainer registry and YAML-backed config helpers."""

from __future__ import annotations

import copy
import itertools
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


_CONFIG_ID_ALIASES: dict[str, dict[str, str]] = {
    "lime": {
        "lime_kernel_width": "kernel",
        "lime_num_samples": "samples",
    },
    "shap": {
        "background_sample_size": "background",
        "shap_explainer_type": "explainer",
        "shap_nsamples": "nsamples",
        "shap_l1_reg": "l1reg",
        "shap_l1_reg_k": "l1regk",
    },
    "causal_shap": {
        "background_sample_size": "background",
        "causal_shap_coalitions": "coalitions",
        "causal_shap_corr_threshold": "corr",
    },
    "integrated_gradients": {
        "ig_steps": "steps",
    },
}


def resolve_explainer_config(
    explainer_name: str,
    config_id: str,
    *,
    registry: ExplainerRegistry | None = None,
) -> dict[str, Any]:
    """Resolve one stable config_id into the concrete explainer override dict."""

    configs = build_explainer_config_registry(explainer_name, registry=registry)
    try:
        resolved = configs[config_id]
    except KeyError as exc:
        supported = ", ".join(sorted(configs))
        raise KeyError(
            f"Unknown config_id '{config_id}' for explainer '{explainer_name}'. "
            f"Supported config_ids: {supported}."
        ) from exc
    return copy.deepcopy(resolved)


def build_explainer_config_registry(
    explainer_name: str,
    *,
    registry: ExplainerRegistry | None = None,
    grid_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the stable config registry for one explainer from the YAML grids."""

    explainer_spec = (registry or DEFAULT_EXPLAINER_REGISTRY).get(explainer_name)
    base_params = copy.deepcopy(explainer_spec.get("params", {}) or {})
    explanation_cfg = copy.deepcopy(base_params.get("experiment", {}).get("explanation", {}) or {})

    raw_grid = load_explainer_hyperparameter_grid(path=grid_path).get(explainer_name) or {}
    list_params = {
        key: list(value)
        for key, value in raw_grid.items()
        if isinstance(value, list) and value
    }
    if not list_params:
        return {explainer_name: explanation_cfg}

    aliases = _CONFIG_ID_ALIASES.get(explainer_name, {})
    ordered_keys = sorted(list_params, key=lambda key: (aliases.get(key, key), key))
    registry_entries: dict[str, dict[str, Any]] = {}
    for combination in itertools.product(*(list_params[key] for key in ordered_keys)):
        overrides = copy.deepcopy(explanation_cfg)
        id_parts = [explainer_name]
        for key, value in zip(ordered_keys, combination, strict=True):
            overrides[key] = value
            alias = aliases.get(key, key)
            id_parts.append(f"{alias}-{_normalize_config_id_value(value)}")
        registry_entries["__".join(id_parts)] = overrides
    return registry_entries


def _normalize_config_id_value(value: Any) -> str:
    text = str(value)
    return text.replace(" ", "-").replace("/", "-").replace("_", "-").lower()
