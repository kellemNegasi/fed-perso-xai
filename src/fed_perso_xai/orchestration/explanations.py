"""Client-local explanation orchestration for federated runs."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fed_perso_xai.data.serialization import load_client_datasets
from fed_perso_xai.explainers import DEFAULT_EXPLAINER_REGISTRY, make_explainer
from fed_perso_xai.fl.client import ClientData
from fed_perso_xai.models import create_model
from fed_perso_xai.utils.config import ArtifactPaths, LogisticRegressionConfig
from fed_perso_xai.utils.paths import centralized_run_dir, federated_run_dir, partition_root


@dataclass(frozen=True)
class LocalExplanationDataset:
    """Minimal dataset view expected by the Perso-XAI-style explainer base class."""

    X_train: np.ndarray
    y_train: np.ndarray
    feature_names: list[str]
    task: str = "classification"
    background_data_source: str = "client_local_train"

    @property
    def feature_means(self) -> np.ndarray:
        return np.mean(self.X_train, axis=0)


class ExplainerModelAdapter:
    """Expose sklearn-like classifier methods around the stage-1 model contract."""

    _estimator_type = "classifier"
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, model: Any):
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.int64)

    def predict_numeric(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = np.asarray(self.model.predict_proba(X), dtype=np.float64).reshape(-1)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])


def instantiate_explainer(
    name: str,
    model: Any,
    dataset: LocalExplanationDataset,
    *,
    logging_cfg: dict[str, Any] | None = None,
    params_override: dict[str, Any] | None = None,
) -> Any:
    """Instantiate and fit one explainer from the YAML-backed registry."""

    spec = DEFAULT_EXPLAINER_REGISTRY.get(name)
    config = {"name": name, "type": spec["type"]}
    params = copy.deepcopy(spec.get("params", {}) or {})
    config.update(params)
    if params_override:
        exp_cfg = config.setdefault("experiment", {})
        expl_cfg = exp_cfg.setdefault("explanation", {})
        expl_cfg.update(params_override)
        config["_override_params"] = copy.deepcopy(params_override)
    if logging_cfg:
        experiment_cfg = config.setdefault("experiment", {})
        current_logging = experiment_cfg.get("logging", {}) or {}
        experiment_cfg["logging"] = {**current_logging, **logging_cfg}

    explainer = make_explainer(
        config=config,
        model=ExplainerModelAdapter(model),
        dataset=dataset,
    )
    explainer.fit(dataset.X_train, dataset.y_train)
    return explainer


def generate_client_local_explanations(
    *,
    client_data: ClientData,
    model: Any,
    feature_names: list[str],
    explainer_name: str = "shap",
    split_name: str = "test",
    logging_cfg: dict[str, Any] | None = None,
    params_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate local explanations for one client using only that client's train split as SHAP background."""

    dataset = LocalExplanationDataset(
        X_train=np.asarray(client_data.X_train, dtype=np.float64),
        y_train=np.asarray(client_data.y_train, dtype=np.int64),
        feature_names=list(feature_names),
    )
    explainer = instantiate_explainer(
        explainer_name,
        model=model,
        dataset=dataset,
        logging_cfg=logging_cfg,
        params_override=params_override,
    )
    X_split, y_split, row_ids = client_data.get_split(split_name)
    result = explainer.explain_dataset(X_split, y_split)

    selected_row_ids = np.asarray(row_ids, dtype=str)
    sample_indices = explainer.sample_indices()
    if sample_indices is not None:
        selected_row_ids = selected_row_ids[sample_indices]

    result["client_id"] = int(client_data.client_id)
    result["split_name"] = split_name
    result["row_ids"] = selected_row_ids.tolist()
    result.setdefault("info", {})
    result["info"]["client_context"] = {
        "client_id": int(client_data.client_id),
        "split_name": split_name,
        "background_data_source": dataset.background_data_source,
        "background_dataset_size": int(dataset.X_train.shape[0]),
    }
    return to_serializable(result)


def load_feature_names_from_metadata(path: Path) -> list[str]:
    """Read transformed feature names from saved feature metadata."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("stable_transformed_feature_order") or payload.get(
        "transformed_feature_names"
    )
    if not isinstance(names, list) or not names:
        raise ValueError(f"Feature metadata at {path} does not expose transformed feature names.")
    return [str(name) for name in names]


def save_client_explanations(path: Path, payload: dict[str, Any]) -> Path:
    """Persist explanations as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")
    return path


def load_client_data_for_explanations(
    *,
    paths: ArtifactPaths,
    dataset_name: str,
    num_clients: int,
    alpha: float,
    seed: int,
    client_id: int,
) -> ClientData:
    """Load one saved client partition for explanation generation."""

    root_dir = partition_root(paths.partition_root, dataset_name, num_clients, alpha, seed)
    datasets = load_client_datasets(root_dir, num_clients)
    try:
        client = next(item for item in datasets if item.client_id == client_id)
    except StopIteration as exc:
        raise ValueError(
            f"Client {client_id} not found under prepared partition root {root_dir}."
        ) from exc

    return ClientData(
        client_id=client.client_id,
        X_train=client.train.X,
        y_train=client.train.y,
        row_ids_train=client.train.row_ids,
        X_test=client.test.X,
        y_test=client.test.y,
        row_ids_test=client.test.row_ids,
    )


def load_saved_model_for_explanations(
    *,
    paths: ArtifactPaths,
    dataset_name: str,
    seed: int,
    model_source: str,
    num_clients: int,
    alpha: float,
) -> tuple[Any, Path]:
    """Load a saved stage-1 model artifact and rebuild the model wrapper."""

    normalized_source = model_source.strip().lower()
    if normalized_source == "federated":
        result_dir = federated_run_dir(paths, dataset_name, num_clients, alpha, seed)
    elif normalized_source == "centralized":
        result_dir = centralized_run_dir(paths, dataset_name, seed)
    else:
        raise ValueError(
            f"Unsupported model source '{model_source}'. Expected 'federated' or 'centralized'."
        )

    config_path = result_dir / "config_snapshot.json"
    model_path = result_dir / "model_parameters.npz"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot at '{config_path}'.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact at '{model_path}'.")

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = str(config_payload.get("model_name", "logistic_regression"))
    model_cfg = LogisticRegressionConfig(**dict(config_payload.get("model", {}) or {}))

    data = np.load(model_path, allow_pickle=False)
    weights = np.asarray(data["weights"], dtype=np.float64)
    bias = np.asarray(data["bias"], dtype=np.float64).reshape(1)
    model = create_model(
        model_name,
        n_features=int(weights.shape[0]),
        config=model_cfg,
    )
    model.set_parameters([weights, bias])
    return model, result_dir


def resolve_feature_names_for_explanations(
    *,
    paths: ArtifactPaths,
    dataset_name: str,
    seed: int,
    model_source: str,
    num_clients: int,
    alpha: float,
) -> tuple[list[str], Path]:
    """Resolve feature metadata from the selected run, with partition fallback."""

    normalized_source = model_source.strip().lower()
    if normalized_source == "federated":
        result_dir = federated_run_dir(paths, dataset_name, num_clients, alpha, seed)
    elif normalized_source == "centralized":
        result_dir = centralized_run_dir(paths, dataset_name, seed)
    else:
        raise ValueError(
            f"Unsupported model source '{model_source}'. Expected 'federated' or 'centralized'."
        )

    candidate = result_dir / "feature_metadata.json"
    if candidate.exists():
        return load_feature_names_from_metadata(candidate), candidate

    partition_metadata_path = (
        partition_root(paths.partition_root, dataset_name, num_clients, alpha, seed)
        / "partition_metadata.json"
    )
    if not partition_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing feature metadata in '{candidate}' and partition metadata in '{partition_metadata_path}'."
        )
    partition_metadata = json.loads(partition_metadata_path.read_text(encoding="utf-8"))
    fallback_path = Path(str(partition_metadata["feature_metadata_path"]))
    return load_feature_names_from_metadata(fallback_path), fallback_path


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    return obj
