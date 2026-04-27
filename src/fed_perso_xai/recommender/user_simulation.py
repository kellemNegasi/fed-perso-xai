"""Simulate user preferences over client-local explanation candidates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import yaml

from fed_perso_xai.orchestration.run_artifacts import resolve_federated_run_context
from fed_perso_xai.recommender.data import split_recommender_instance_ids
from fed_perso_xai.utils.config import ArtifactPaths
from fed_perso_xai.utils.provenance import current_utc_timestamp


DEFAULT_TAU = 0.05
MIN_DIRICHLET_CONCENTRATION = 1e-6
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_PREFERENCE_MODEL_PATH = DEFAULT_CONFIG_DIR / "preference_model.yml"


@dataclass(frozen=True)
class PropertyConfig:
    """A persona property and its covered explanation metrics."""

    name: str
    preference: float
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class PersonaConfig:
    """A simulated-user persona defined by weighted metric groups."""

    persona: str
    type: str
    description: str | None
    tau: float | None
    properties: tuple[PropertyConfig, ...]

    def metric_names(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for prop in self.properties:
            for metric_name in prop.metrics:
                if metric_name not in seen:
                    ordered.append(metric_name)
                    seen.add(metric_name)
        return tuple(ordered)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PersonaConfig":
        persona = payload.get("persona")
        persona_type = payload.get("type")
        if not isinstance(persona, str) or not persona:
            raise ValueError("Persona config missing non-empty 'persona' field.")
        if not isinstance(persona_type, str) or not persona_type:
            raise ValueError("Persona config missing non-empty 'type' field.")
        description = payload.get("description")
        if description is not None and not isinstance(description, str):
            description = None
        raw_tau = payload.get("tau")
        if raw_tau is None:
            tau = None
        elif isinstance(raw_tau, (int, float)) and float(raw_tau) > 0:
            tau = float(raw_tau)
        else:
            raise ValueError("Persona config 'tau' must be a positive number when provided.")

        raw_properties = payload.get("properties")
        if not isinstance(raw_properties, Mapping) or not raw_properties:
            raise ValueError("Persona config missing non-empty 'properties' mapping.")

        properties: list[PropertyConfig] = []
        seen_metrics: set[str] = set()
        for prop_name, prop_payload in raw_properties.items():
            if not isinstance(prop_name, str) or not isinstance(prop_payload, Mapping):
                continue
            preference = prop_payload.get("preference")
            metrics = prop_payload.get("metrics")
            if not isinstance(preference, (int, float)) or float(preference) <= 0:
                raise ValueError(f"Property {prop_name!r} has invalid preference={preference!r}.")
            if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)) or not metrics:
                raise ValueError(f"Property {prop_name!r} must define a non-empty metrics list.")

            metric_names: list[str] = []
            for metric_name in metrics:
                if not isinstance(metric_name, str) or not metric_name:
                    raise ValueError(f"Property {prop_name!r} has invalid metric name={metric_name!r}.")
                if metric_name in seen_metrics:
                    raise ValueError(f"Metric {metric_name!r} is listed under multiple properties.")
                metric_names.append(metric_name)
                seen_metrics.add(metric_name)
            properties.append(
                PropertyConfig(
                    name=prop_name,
                    preference=float(preference),
                    metrics=tuple(metric_names),
                )
            )
        if not properties:
            raise ValueError("Persona config contains no valid properties.")
        return cls(
            persona=persona,
            type=persona_type,
            description=description,
            tau=tau,
            properties=tuple(properties),
        )


@dataclass(frozen=True)
class PairwiseLabelConfig:
    """Column names used for pairwise preference-label artifacts."""

    dataset_index: str = "dataset_index"
    pair_1: str = "pair_1"
    pair_2: str = "pair_2"
    label: str = "label"
    probability_pair_1_preferred: str = "probability_pair_1_preferred"
    utility_pair_1: str = "utility_pair_1"
    utility_pair_2: str = "utility_pair_2"


class UserSimulator(Protocol):
    """Protocol for replaceable user simulators."""

    @property
    def name(self) -> str:
        """Simulator type name."""

    def label_client_candidates(self, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return pairwise labels and simulator metadata for one client's candidates."""


SimulatorFactory = Callable[..., UserSimulator]


class UserSimulatorRegistry:
    """Small registry so future simulators can be swapped by name."""

    def __init__(self) -> None:
        self._factories: dict[str, SimulatorFactory] = {}

    def register(self, name: str, factory: SimulatorFactory) -> None:
        if not name:
            raise ValueError("Simulator name must be non-empty.")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> UserSimulator:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._factories))
            raise KeyError(f"Unknown user simulator {name!r}. Available: {choices}") from exc
        return factory(**kwargs)

    def list_keys(self) -> list[str]:
        return sorted(self._factories)


class DirichletPersonaSimulator:
    """Flat Dirichlet persona that labels all candidate pairs per instance."""

    name = "dirichlet_persona"

    def __init__(
        self,
        persona_config: PersonaConfig,
        *,
        seed: int | None = None,
        label_seed: int | None = None,
        tau: float | None = None,
        concentration_c: float | None = None,
        preference_model_path: Path = DEFAULT_PREFERENCE_MODEL_PATH,
        columns: PairwiseLabelConfig | None = None,
    ) -> None:
        resolved_tau = persona_config.tau if tau is None and persona_config.tau is not None else tau
        self.tau = float(DEFAULT_TAU if resolved_tau is None else resolved_tau)
        if self.tau <= 0:
            raise ValueError("tau must be > 0.")
        self.persona_config = persona_config
        self.columns = columns or PairwiseLabelConfig()
        self._weight_rng = np.random.default_rng(seed)
        self._label_rng = np.random.default_rng(label_seed if label_seed is not None else seed)
        self.concentration_c = _resolve_concentration(
            persona=persona_config.persona,
            override=concentration_c,
            preference_model_path=preference_model_path,
        )
        self.metric_weights = self._sample_metric_weights()

    def label_client_candidates(self, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        _validate_candidate_frame(candidates)
        metric_columns = _resolve_metric_z_columns(candidates, self.persona_config.metric_names())
        if not metric_columns:
            raise ValueError(
                "No persona metrics are present in candidate context. Expected columns like "
                "'metric_<metric_name>_z'."
            )

        active_metrics = tuple(metric_columns)
        active_weights = np.asarray([self.metric_weights[name] for name in active_metrics], dtype=float)
        weight_sum = float(active_weights.sum())
        if weight_sum <= 0 or not math.isfinite(weight_sum):
            raise ValueError("Active persona metric weights sum to zero.")
        active_weights = active_weights / weight_sum

        rows: list[dict[str, Any]] = []
        for dataset_index, group in candidates.groupby("dataset_index", sort=True, dropna=False):
            group = _dedupe_candidates(group).reset_index(drop=True)
            if len(group) < 2:
                continue
            z = _metric_matrix(group, active_metrics)
            utilities = z @ active_weights
            variants = group["method_variant"].astype(str).tolist()
            for idx_a, idx_b in combinations(range(len(group)), 2):
                logit = float((utilities[idx_a] - utilities[idx_b]) / self.tau)
                probability = _sigmoid_scalar(logit)
                pair_1_preferred = bool(self._label_rng.random() < probability)
                rows.append(
                    {
                        "client_id": str(group.iloc[0].get("client_id", "")),
                        "instance_id": str(group.iloc[0].get("instance_id", "")),
                        self.columns.dataset_index: _coerce_optional_int(dataset_index),
                        self.columns.pair_1: variants[idx_a],
                        self.columns.pair_2: variants[idx_b],
                        self.columns.label: 0 if pair_1_preferred else 1,
                        self.columns.probability_pair_1_preferred: probability,
                        self.columns.utility_pair_1: float(utilities[idx_a]),
                        self.columns.utility_pair_2: float(utilities[idx_b]),
                    }
                )

        label_frame = pd.DataFrame(rows)
        metadata = {
            "simulator": self.name,
            "persona": self.persona_config.persona,
            "persona_type": self.persona_config.type,
            "tau": self.tau,
            "concentration_c": self.concentration_c,
            "configured_metric_count": len(self.persona_config.metric_names()),
            "active_metrics": list(active_metrics),
            "missing_configured_metrics": [
                name for name in self.persona_config.metric_names() if name not in set(active_metrics)
            ],
            "metric_weights": {
                name: float(self.metric_weights[name])
                for name in self.persona_config.metric_names()
            },
            "active_metric_weights": {
                name: float(weight) for name, weight in zip(active_metrics, active_weights)
            },
            "pair_count": int(len(label_frame)),
        }
        return label_frame, metadata

    def _sample_metric_weights(self) -> dict[str, float]:
        metric_names: list[str] = []
        metric_scores: list[float] = []
        for prop in self.persona_config.properties:
            for metric_name in prop.metrics:
                metric_names.append(metric_name)
                metric_scores.append(prop.preference)
        base = _normalize_positive(metric_scores, label="Metric preference")
        alphas = np.maximum(self.concentration_c * base, MIN_DIRICHLET_CONCENTRATION)
        sampled = self._weight_rng.dirichlet(alphas)
        return {name: float(weight) for name, weight in zip(metric_names, sampled)}


def load_persona_config(path: Path) -> PersonaConfig:
    """Load a persona YAML config."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Persona config must be a YAML mapping. Got: {type(payload)}")
    return PersonaConfig.from_dict(payload)


def default_persona_config_path(persona: str) -> Path:
    """Resolve a bundled persona config by persona name."""

    candidate = DEFAULT_CONFIG_DIR / f"{persona}.yaml"
    if not candidate.exists():
        choices = sorted(path.stem for path in DEFAULT_CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No bundled persona config for {persona!r}. Available: {choices}")
    return candidate


def label_recommender_context(
    *,
    run_id: str,
    selection_id: str,
    persona: str = "lay",
    persona_config_path: Path | None = None,
    simulator: str = "dirichlet_persona",
    clients: str = "all",
    context_filename: str = "candidate_context.parquet",
    label_filename: str = "pairwise_labels.parquet",
    seed: int = 42,
    label_seed: int = 1729,
    instance_test_size: float = 0.2,
    instance_split_seed: int | None = None,
    tau: float | None = None,
    concentration_c: float | None = None,
    paths: ArtifactPaths | None = None,
) -> dict[str, Any]:
    """Label each selected client's candidate contexts with pairwise simulated preferences."""

    _require_safe_segment(run_id, label="run_id")
    _require_safe_segment(selection_id, label="selection_id")
    _require_safe_segment(label_filename, label="label_filename")
    artifact_paths = paths or ArtifactPaths()
    run_context = resolve_federated_run_context(paths=artifact_paths, run_id=run_id)
    requested_clients = _split_selector(clients)

    persona_path = persona_config_path or default_persona_config_path(persona)
    persona_config = load_persona_config(persona_path)
    if persona != persona_config.persona and persona_config_path is None:
        raise ValueError(
            f"Bundled persona config mismatch: requested {persona!r}, "
            f"got {persona_config.persona!r}."
        )

    context_root = run_context.run_artifact_dir / "clients"
    client_dirs = [
        path for path in sorted(context_root.glob("client_*")) if path.is_dir()
    ]
    if requested_clients is not None:
        client_dirs = [path for path in client_dirs if path.name in requested_clients]
        missing = sorted(requested_clients - {path.name for path in client_dirs})
        if missing:
            raise FileNotFoundError(f"Requested clients do not have run directories: {missing}")
    if not client_dirs:
        raise FileNotFoundError(f"No client directories found under {context_root}.")

    output_root = run_context.run_artifact_dir / "recommender_labels" / selection_id / persona_config.persona
    client_summaries: list[dict[str, Any]] = []
    for client_dir in client_dirs:
        client_id = client_dir.name
        context_path = client_dir / "recommender_context" / selection_id / context_filename
        if not context_path.exists():
            if requested_clients is not None:
                raise FileNotFoundError(f"Missing recommender context for {client_id}: {context_path}")
            continue
        candidates = pd.read_parquet(context_path)
        if candidates.empty:
            continue
        base_split_seed = seed if instance_split_seed is None else instance_split_seed
        client_split_seed = _stable_client_seed(
            base_seed=base_split_seed,
            client_id=client_id,
            purpose="instance_split",
        )
        instance_split = split_recommender_instance_ids(
            candidates,
            test_size=instance_test_size,
            random_state=client_split_seed,
        )
        train_candidates = candidates.loc[
            candidates["dataset_index"].isin(instance_split.train_instance_ids)
        ].copy()
        test_candidates = candidates.loc[
            candidates["dataset_index"].isin(instance_split.test_instance_ids)
        ].copy()
        client_seed = _stable_client_seed(base_seed=seed, client_id=client_id, purpose="weights")
        client_label_seed = _stable_client_seed(
            base_seed=label_seed,
            client_id=client_id,
            purpose="labels",
        )
        simulator_instance = DEFAULT_USER_SIMULATOR_REGISTRY.create(
            simulator,
            persona_config=persona_config,
            seed=client_seed,
            label_seed=client_label_seed,
            tau=tau,
            concentration_c=concentration_c,
        )
        train_labels, simulator_metadata = simulator_instance.label_client_candidates(train_candidates)
        test_labels, _ = simulator_instance.label_client_candidates(test_candidates)
        if not train_labels.empty:
            train_labels = train_labels.assign(split="train")
        if not test_labels.empty:
            test_labels = test_labels.assign(split="test")
        labels = pd.concat([train_labels, test_labels], ignore_index=True)

        client_label_dir = client_dir / "recommender_labels" / selection_id / persona_config.persona
        labels_path = client_label_dir / label_filename
        metadata_path = client_label_dir / "simulation_metadata.json"
        _write_parquet_atomic(labels_path, labels)
        client_metadata = {
            "run_id": run_id,
            "selection_id": selection_id,
            "client_id": client_id,
            "persona": persona_config.persona,
            "simulator": simulator,
            "context_path": str(context_path),
            "pairwise_labels": str(labels_path),
            "seed": client_seed,
            "label_seed": client_label_seed,
            "instance_split_seed": client_split_seed,
            "candidate_count": int(len(candidates)),
            "instance_count": int(candidates["dataset_index"].nunique()),
            "pair_count": int(len(labels)),
            "train_pair_count": int(len(train_labels)),
            "test_pair_count": int(len(test_labels)),
            "generated_at": current_utc_timestamp(),
            "instance_split": {
                "strategy": "dataset_index_train_test_split",
                "test_size": float(instance_test_size),
                "random_state": int(client_split_seed),
                "train_dataset_indices": [int(value) for value in instance_split.train_instance_ids],
                "test_dataset_indices": [int(value) for value in instance_split.test_instance_ids],
                "train_instance_count": int(len(instance_split.train_instance_ids)),
                "test_instance_count": int(len(instance_split.test_instance_ids)),
            },
            "simulation": simulator_metadata,
        }
        _write_json_atomic(metadata_path, client_metadata)
        client_summaries.append(
            {
                "client_id": client_id,
                "candidate_count": client_metadata["candidate_count"],
                "instance_count": client_metadata["instance_count"],
                "pair_count": client_metadata["pair_count"],
                "train_pair_count": client_metadata["train_pair_count"],
                "test_pair_count": client_metadata["test_pair_count"],
                "artifacts": {
                    "pairwise_labels": str(labels_path),
                    "simulation_metadata": str(metadata_path),
                },
            }
        )

    if not client_summaries:
        raise FileNotFoundError(
            "No client recommender context files were found. Run prepare-recommender-context first."
        )

    manifest = {
        "status": "labeled",
        "run_id": run_id,
        "selection_id": selection_id,
        "persona": persona_config.persona,
        "persona_config_path": str(persona_path),
        "persona_config_sha256": _sha256_file(persona_path),
        "simulator": simulator,
        "context_filename": context_filename,
        "label_filename": label_filename,
        "client_count": len(client_summaries),
        "candidate_count": int(sum(item["candidate_count"] for item in client_summaries)),
        "pair_count": int(sum(item["pair_count"] for item in client_summaries)),
        "train_pair_count": int(sum(item["train_pair_count"] for item in client_summaries)),
        "test_pair_count": int(sum(item["test_pair_count"] for item in client_summaries)),
        "generated_at": current_utc_timestamp(),
        "clients": client_summaries,
    }
    manifest_path = output_root / "labeling_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _resolve_concentration(
    *,
    persona: str,
    override: float | None,
    preference_model_path: Path,
) -> float:
    if override is not None:
        value = float(override)
    else:
        raw = yaml.safe_load(preference_model_path.read_text(encoding="utf-8"))
        payload = raw.get("preference_model") if isinstance(raw, Mapping) else None
        concentration = payload.get("concentration") if isinstance(payload, Mapping) else None
        fixed = concentration.get("fixed") if isinstance(concentration, Mapping) else None
        value = fixed.get(persona) if isinstance(fixed, Mapping) else None
        if not isinstance(value, (int, float)):
            raise ValueError(f"Missing positive concentration for persona={persona!r}.")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("concentration_c must be > 0.")
    return float(max(float(value), MIN_DIRICHLET_CONCENTRATION))


def _resolve_metric_z_columns(candidates: pd.DataFrame, metric_names: Sequence[str]) -> tuple[str, ...]:
    available = set(candidates.columns)
    resolved: list[str] = []
    for metric_name in metric_names:
        z_col = f"metric_{metric_name}_z"
        if z_col in available:
            resolved.append(metric_name)
    return tuple(resolved)


def _metric_matrix(candidates: pd.DataFrame, metric_names: Sequence[str]) -> np.ndarray:
    columns = [f"metric_{name}_z" for name in metric_names]
    return (
        candidates.loc[:, columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _dedupe_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates["method_variant"].astype(str).is_unique:
        return candidates
    numeric_cols = candidates.select_dtypes(include=["number", "bool"]).columns.tolist()
    first_cols = [col for col in candidates.columns if col not in numeric_cols and col != "method_variant"]
    grouped = candidates.assign(method_variant=candidates["method_variant"].astype(str)).groupby(
        "method_variant",
        sort=False,
    )
    parts: list[pd.DataFrame] = []
    if first_cols:
        parts.append(grouped[first_cols].first())
    if numeric_cols:
        parts.append(grouped[numeric_cols].mean())
    return pd.concat(parts, axis=1).reset_index()


def _validate_candidate_frame(candidates: pd.DataFrame) -> None:
    required = {"dataset_index", "method_variant"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate context is missing required columns: {sorted(missing)}")


def _normalize_positive(scores: Sequence[float], *, label: str) -> np.ndarray:
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"{label} scores must be finite and > 0.")
    return arr / float(arr.sum())


def _sigmoid_scalar(x: float) -> float:
    if x >= 0:
        return float(1.0 / (1.0 + math.exp(-x)))
    exp_x = math.exp(x)
    return float(exp_x / (1.0 + exp_x))


def _split_selector(value: str) -> set[str] | None:
    text = str(value or "all").strip()
    if text.lower() == "all":
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def _stable_client_seed(*, base_seed: int, client_id: str, purpose: str) -> int:
    seed_payload = json.dumps(
        {
            "base_seed": int(base_seed),
            "client_id": str(client_id),
            "purpose": str(purpose),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(seed_payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _require_safe_segment(value: str, *, label: str) -> None:
    segment = str(value)
    if not segment or segment in {".", ".."} or "/" in segment or "\\" in segment:
        raise ValueError(f"{label} must be a single non-empty path segment.")


def _coerce_optional_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


DEFAULT_USER_SIMULATOR_REGISTRY = UserSimulatorRegistry()
DEFAULT_USER_SIMULATOR_REGISTRY.register("dirichlet_persona", DirichletPersonaSimulator)
