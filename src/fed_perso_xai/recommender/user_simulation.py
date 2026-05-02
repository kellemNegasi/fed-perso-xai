"""Simulate user preferences over client-local explanation candidates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
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
DEFAULT_HETEROGENEOUS_PERSONAS = ("lay", "clinician", "regulator")
DEFAULT_PERSONA_ASSIGNMENT_POLICY = "fixed"
DIRICHLET_SAMPLED_PERSONA_ASSIGNMENT_POLICY = "dirichlet_sampled"
DEFAULT_PERSONA_ASSIGNMENT_ALPHA = 0.3
DEFAULT_HETEROGENEOUS_OUTPUT_PERSONA = DIRICHLET_SAMPLED_PERSONA_ASSIGNMENT_POLICY
SUPPORTED_PERSONA_ASSIGNMENT_POLICIES = (
    DEFAULT_PERSONA_ASSIGNMENT_POLICY,
    DIRICHLET_SAMPLED_PERSONA_ASSIGNMENT_POLICY,
)


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
        metric_columns = _resolve_metric_columns(candidates, self.persona_config.metric_names())
        if not metric_columns:
            raise ValueError(
                "No persona metrics are present in candidate context. Expected metric columns like "
                "'<metric_name>' or legacy 'metric_<metric_name>_z'."
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


def assign_dirichlet_personas_to_clients(
    client_ids: Sequence[str],
    users_by_client: Mapping[str, Sequence[str]],
    *,
    alpha: float = DEFAULT_PERSONA_ASSIGNMENT_ALPHA,
    seed: int = 42,
    output_path: Path | None = None,
    personas: Sequence[str] = DEFAULT_HETEROGENEOUS_PERSONAS,
) -> dict[str, Any]:
    """Assign one persona per client from a shared Dirichlet client-mixture."""

    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be > 0.")
    persona_names = tuple(str(persona) for persona in personas)
    if not persona_names:
        raise ValueError("personas must be non-empty.")
    if len(set(persona_names)) != len(persona_names):
        raise ValueError("personas must be unique.")

    artifact = {
        "seed": int(seed),
        "alpha": float(alpha),
        "personas": list(persona_names),
        "assignment_policy": DIRICHLET_SAMPLED_PERSONA_ASSIGNMENT_POLICY,
        "assignment_scope": "client",
        "persona_probs": {},
        "clients": {},
    }
    user_persona_by_client: dict[str, dict[str, str]] = {}
    sorted_client_ids = sorted(str(client_id) for client_id in client_ids)
    rng = np.random.default_rng(int(seed))
    persona_probs = rng.dirichlet(np.full(len(persona_names), float(alpha), dtype=float))
    sampled_client_personas = rng.choice(
        persona_names,
        size=len(sorted_client_ids),
        p=persona_probs,
    )
    artifact["persona_probs"] = {
        persona: float(prob)
        for persona, prob in zip(persona_names, persona_probs.tolist(), strict=True)
    }
    argmax_persona = str(persona_names[int(np.argmax(persona_probs))])

    for client_id, sampled_client_persona in zip(
        sorted_client_ids,
        sampled_client_personas.tolist(),
        strict=True,
    ):
        users = [str(user_id) for user_id in users_by_client.get(client_id, ())]
        sampled_client_persona = str(sampled_client_persona)
        user_persona_by_client[client_id] = {
            user_id: sampled_client_persona
            for user_id in users
        }
        artifact["clients"][client_id] = {
            "persona_probs": {
                persona: float(prob)
                for persona, prob in zip(persona_names, persona_probs.tolist(), strict=True)
            },
            "sampled_client_persona": sampled_client_persona,
            "argmax_persona": argmax_persona,
            "user_persona_counts": {
                persona: int(len(users) if persona == sampled_client_persona else 0)
                for persona in persona_names
            },
        }

    if output_path is not None:
        _write_json_atomic(output_path, artifact)

    return {
        "artifact": artifact,
        "user_persona_by_client": user_persona_by_client,
    }


def label_recommender_context(
    *,
    run_id: str,
    selection_id: str,
    persona: str = "lay",
    persona_config_path: Path | None = None,
    output_persona: str | None = None,
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
    persona_assignment_policy: str = DEFAULT_PERSONA_ASSIGNMENT_POLICY,
    persona_assignment_alpha: float = DEFAULT_PERSONA_ASSIGNMENT_ALPHA,
    paths: ArtifactPaths | None = None,
) -> dict[str, Any]:
    """Label each selected client's candidate contexts with pairwise simulated preferences."""

    _require_safe_segment(run_id, label="run_id")
    _require_safe_segment(selection_id, label="selection_id")
    _require_safe_segment(context_filename, label="context_filename")
    _require_safe_segment(label_filename, label="label_filename")
    artifact_paths = paths or ArtifactPaths()
    run_context = resolve_federated_run_context(paths=artifact_paths, run_id=run_id)
    requested_clients = _split_selector(clients)
    if persona_assignment_policy not in SUPPORTED_PERSONA_ASSIGNMENT_POLICIES:
        choices = ", ".join(sorted(SUPPORTED_PERSONA_ASSIGNMENT_POLICIES))
        raise ValueError(
            f"Unsupported persona_assignment_policy={persona_assignment_policy!r}. Available: {choices}."
        )

    if persona_assignment_policy == DEFAULT_PERSONA_ASSIGNMENT_POLICY:
        persona_path = persona_config_path or default_persona_config_path(persona)
        persona_config = load_persona_config(persona_path)
        _require_safe_segment(persona_config.persona, label="persona")
        if persona != persona_config.persona:
            source_label = "Bundled" if persona_config_path is None else "Custom"
            raise ValueError(
                f"{source_label} persona config mismatch: requested {persona!r}, "
                f"got {persona_config.persona!r}."
            )
        output_persona_name = output_persona or persona_config.persona
        _require_safe_segment(output_persona_name, label="output_persona")
        persona_configs = {persona_config.persona: persona_config}
    else:
        if persona_config_path is not None:
            raise ValueError("persona_config_path is not supported with dirichlet_sampled assignment.")
        persona_path = None
        output_persona_name = output_persona or DEFAULT_HETEROGENEOUS_OUTPUT_PERSONA
        _require_safe_segment(output_persona_name, label="output_persona")
        persona_configs = {
            persona_name: load_persona_config(default_persona_config_path(persona_name))
            for persona_name in DEFAULT_HETEROGENEOUS_PERSONAS
        }

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

    output_root = run_context.run_artifact_dir / "recommender_labels" / selection_id / output_persona_name
    manifest_path = output_root / "labeling_manifest.json"
    assignment_path = output_root / "persona_assignment.json"
    client_summaries: list[dict[str, Any]] = []
    found_candidate_context = False
    client_candidates: dict[str, pd.DataFrame] = {}
    client_user_ids: dict[str, list[str]] = {}
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
        found_candidate_context = True
        client_candidates[client_id] = candidates
        client_user_ids[client_id] = _resolve_client_user_ids(candidates)
    if not found_candidate_context:
        raise FileNotFoundError(
            "No client recommender context files were found. Run prepare-recommender-context first."
        )

    persona_assignment_artifact: dict[str, Any] | None = None
    user_persona_by_client: dict[str, dict[str, str]] = {}
    if persona_assignment_policy == DIRICHLET_SAMPLED_PERSONA_ASSIGNMENT_POLICY:
        assignment = assign_dirichlet_personas_to_clients(
            sorted(client_candidates),
            client_user_ids,
            alpha=persona_assignment_alpha,
            seed=seed,
            output_path=assignment_path,
            personas=tuple(persona_configs),
        )
        persona_assignment_artifact = dict(assignment["artifact"])
        user_persona_by_client = {
            str(client_id): {
                str(user_id): str(persona_name)
                for user_id, persona_name in mapping.items()
            }
            for client_id, mapping in assignment["user_persona_by_client"].items()
        }

    for client_dir in client_dirs:
        client_id = client_dir.name
        candidates = client_candidates.get(client_id)
        if candidates is None:
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
        if persona_assignment_policy == DEFAULT_PERSONA_ASSIGNMENT_POLICY:
            fixed_persona = next(iter(persona_configs.values()))
            client_seed = _stable_client_seed(base_seed=seed, client_id=client_id, purpose="weights")
            client_label_seed = _stable_client_seed(
                base_seed=label_seed,
                client_id=client_id,
                purpose="labels",
            )
            simulator_instance = DEFAULT_USER_SIMULATOR_REGISTRY.create(
                simulator,
                persona_config=fixed_persona,
                seed=client_seed,
                label_seed=client_label_seed,
                tau=tau,
                concentration_c=concentration_c,
            )
            train_labels, simulator_metadata = simulator_instance.label_client_candidates(train_candidates)
            test_labels, _ = simulator_instance.label_client_candidates(test_candidates)
            persona_metadata = None
            effective_persona_name = fixed_persona.persona
            metadata_seed = int(client_seed)
            metadata_label_seed = int(client_label_seed)
        else:
            client_persona = _resolve_client_persona(
                client_id=client_id,
                user_personas=user_persona_by_client.get(client_id, {}),
            )
            persona_seeds = {
                "seed": _stable_client_seed(
                    base_seed=seed,
                    client_id=client_id,
                    purpose=f"weights:{client_persona}",
                ),
                "label_seed": _stable_client_seed(
                    base_seed=label_seed,
                    client_id=client_id,
                    purpose=f"labels:{client_persona}",
                ),
            }
            simulator_instance = DEFAULT_USER_SIMULATOR_REGISTRY.create(
                simulator,
                persona_config=persona_configs[client_persona],
                seed=persona_seeds["seed"],
                label_seed=persona_seeds["label_seed"],
                tau=tau,
                concentration_c=concentration_c,
            )
            train_labels, train_metadata = simulator_instance.label_client_candidates(train_candidates)
            test_labels, test_metadata = simulator_instance.label_client_candidates(test_candidates)
            simulator_metadata = {
                "simulator": simulator,
                "assignment_policy": persona_assignment_policy,
                "assignment_alpha": float(persona_assignment_alpha),
                "persona_seeds": persona_seeds,
                "sampled_client_persona": client_persona,
                "persona": _merge_persona_simulation_metadata(train_metadata, test_metadata),
            }
            persona_metadata = persona_assignment_artifact["clients"][client_id] if persona_assignment_artifact else None
            effective_persona_name = output_persona_name
            metadata_seed = int(seed)
            metadata_label_seed = int(label_seed)
        if not train_labels.empty:
            if persona_assignment_policy == DIRICHLET_SAMPLED_PERSONA_ASSIGNMENT_POLICY:
                train_labels = train_labels.assign(assigned_persona=client_persona)
            train_labels = train_labels.assign(split="train")
        if not test_labels.empty:
            if persona_assignment_policy == DIRICHLET_SAMPLED_PERSONA_ASSIGNMENT_POLICY:
                test_labels = test_labels.assign(assigned_persona=client_persona)
            test_labels = test_labels.assign(split="test")
        labels = pd.concat([train_labels, test_labels], ignore_index=True)
        if labels.empty:
            continue

        client_label_dir = client_dir / "recommender_labels" / selection_id / output_persona_name
        labels_path = client_label_dir / label_filename
        metadata_path = client_label_dir / "simulation_metadata.json"
        _write_parquet_atomic(labels_path, labels)
        client_metadata = {
            "run_id": run_id,
            "selection_id": selection_id,
            "client_id": client_id,
            "persona": effective_persona_name,
            "simulator": simulator,
            "context_path": str(client_dir / "recommender_context" / selection_id / context_filename),
            "pairwise_labels": str(labels_path),
            "seed": metadata_seed,
            "label_seed": metadata_label_seed,
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
        if persona_metadata is not None:
            client_metadata["persona_assignment"] = persona_metadata
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
        if manifest_path.exists():
            manifest_path.unlink()
        if assignment_path.exists():
            assignment_path.unlink()
        raise FileNotFoundError(
            "No recommender preference pairs were generated. Ensure at least one client has "
            "two or more candidates per instance after filtering."
        )

    manifest = {
        "status": "labeled",
        "run_id": run_id,
        "selection_id": selection_id,
        "persona": output_persona_name,
        "persona_assignment_policy": persona_assignment_policy,
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
    if persona_path is not None:
        manifest["persona_config_path"] = str(persona_path)
        manifest["persona_config_sha256"] = _sha256_file(persona_path)
    if persona_assignment_artifact is not None:
        manifest["persona_assignment"] = {
            "artifact_path": str(assignment_path),
            "alpha": float(persona_assignment_alpha),
            "personas": list(persona_assignment_artifact["personas"]),
            "assignment_policy": persona_assignment_artifact["assignment_policy"],
        }
    _write_json_atomic(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _resolve_client_user_ids(candidates: pd.DataFrame) -> list[str]:
    user_ids: list[str] = []
    for dataset_index in sorted(candidates["dataset_index"].dropna().unique().tolist()):
        resolved_index = _coerce_optional_int(dataset_index)
        if resolved_index is None:
            raise ValueError("Candidate context contains a non-integer dataset_index.")
        user_ids.append(str(resolved_index))
    return user_ids


def _resolve_client_persona(
    *,
    client_id: str,
    user_personas: Mapping[str, str],
) -> str:
    personas = sorted({str(persona) for persona in user_personas.values()})
    if not personas:
        raise ValueError(f"Missing persona assignment for client_id={client_id!r}.")
    if len(personas) != 1:
        raise ValueError(
            f"Expected a single client-level persona for client_id={client_id!r}, got {personas}."
        )
    return personas[0]


def _merge_persona_simulation_metadata(
    train_metadata: Mapping[str, Any],
    test_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(test_metadata or train_metadata)
    payload["pair_count"] = int(train_metadata.get("pair_count", 0)) + int(test_metadata.get("pair_count", 0))
    return payload


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


def _resolve_metric_columns(candidates: pd.DataFrame, metric_names: Sequence[str]) -> tuple[str, ...]:
    available = set(candidates.columns)
    resolved: list[str] = []
    for metric_name in metric_names:
        if metric_name in available or f"metric_{metric_name}_z" in available:
            resolved.append(metric_name)
    return tuple(resolved)


def _metric_matrix(candidates: pd.DataFrame, metric_names: Sequence[str]) -> np.ndarray:
    columns = [
        name if name in candidates.columns else f"metric_{name}_z"
        for name in metric_names
    ]
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
