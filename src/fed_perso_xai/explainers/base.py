"""
Base explainer class for local (per-instance) XAI methods on tabular data.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pandas as pd

    _HAS_PANDAS = True
except Exception:  # pragma: no cover - pandas is a project dependency
    logging.getLogger(__name__).debug("pandas not available, skipping related support.")
    _HAS_PANDAS = False


ArrayLike = Union[np.ndarray, "pd.Series", "pd.DataFrame", List[float], Tuple[float, ...]]  # noqa: F821
InstanceLike = Union[np.ndarray, "pd.Series", List[float], Tuple[float, ...]]  # noqa: F821


class BaseExplainer(ABC):
    """
    Base class for all local (per-instance) explanation methods on tabular data.

    Subclasses MUST implement:
        - explain_instance(self, instance: InstanceLike) -> Dict[str, Any]
    """

    supported_data_types: List[str] = ["tabular"]
    supported_model_types: List[str] = ["sklearn", "xgboost", "lightgbm", "catboost", "generic-predict"]

    def __init__(self, config: Dict[str, Any], model: Any, dataset: Any):
        self.config = config or {}
        self.model = model
        self.dataset = dataset

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.generation_time: float = 0.0

        self._exp_cfg = self.config.get("experiment", {}) or {}
        self._expl_cfg = self._exp_cfg.get("explanation", {}) or {}
        self._log_cfg = self._exp_cfg.get("logging", {}) or {}
        log_level = self._log_cfg.get("level")
        if log_level:
            numeric_level = getattr(logging, str(log_level).upper(), None)
            if isinstance(numeric_level, int):
                self.logger.setLevel(numeric_level)
        self._log_progress = bool(
            self.config.get("log_progress")
            or self._exp_cfg.get("log_progress")
            or self._log_cfg.get("progress")
        )

        self.random_state: Optional[int] = self._expl_cfg.get("random_state")
        self._sampling_info: Dict[str, Any] = {
            "strategy": self._expl_cfg.get("sampling_strategy", "sequential"),
            "max_instances": self._expl_cfg.get("max_instances")
            or self._expl_cfg.get("max_test_samples"),
            "method_cap": self._expl_cfg.get("method_max_instances"),
            "original_size": None,
            "selected_size": None,
            "problem_type": None,
        }
        self._sample_indices: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        return None

    @abstractmethod
    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        raise NotImplementedError

    def explain_batch(self, X: ArrayLike) -> List[Dict[str, Any]]:
        X_np, _ = self._coerce_X_y(X, None)
        results: List[Dict[str, Any]] = []
        for i in range(len(X_np)):
            start = time.time()
            res = self.explain_instance(self._row_to_instance(X, i))
            res["generation_time_total"] = time.time() - start
            results.append(res)
        return results

    def explain_dataset(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> Dict[str, Any]:
        start_all = time.time()

        X_np, y_np = self._coerce_X_y(X, y)
        X_np, y_np = self._limit_samples(X_np, y_np)

        if self._log_progress:
            self.logger.info(
                "Running %s explanations on %d instances",
                self.config.get("type", self.__class__.__name__),
                len(X_np),
            )

        explanations = self.explain_batch(X_np)
        explanations = self._augment_metadata(explanations, y_np)

        total_time = time.time() - start_all
        self.generation_time = total_time

        if self._log_progress:
            self.logger.info(
                "Finished %s explanations in %.2fs",
                self.config.get("type", self.__class__.__name__),
                total_time,
            )

        return {
            "method": self.config.get("type", self.__class__.__name__),
            "explanations": explanations,
            "n_explanations": len(explanations),
            "generation_time": total_time,
            "info": self.get_info(),
        }

    def is_compatible(self) -> bool:
        return bool(hasattr(self.model, "predict"))

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.config.get("name", self.__class__.__name__),
            "type": self.config.get("type", "local"),
            "config": self.config,
            "supported_data_types": getattr(self, "supported_data_types", []),
            "supported_model_types": getattr(self, "supported_model_types", []),
            "sampling": self._sampling_info,
        }

    def empty_result(self, reason: str = "Not compatible or failed.") -> Dict[str, Any]:
        return {
            "method": self.config.get("type", self.__class__.__name__),
            "explanations": [],
            "n_explanations": 0,
            "generation_time": 0.0,
            "info": {
                "error": reason,
                "supported_data_types": getattr(self, "supported_data_types", []),
                "supported_model_types": getattr(self, "supported_model_types", []),
            },
        }

    def sample_indices(self) -> np.ndarray | None:
        if self._sample_indices is None:
            return None
        return self._sample_indices.copy()

    def _limit_samples(
        self, X: np.ndarray, y: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        max_n = self._expl_cfg.get("max_instances")
        if max_n is None:
            max_n = self._expl_cfg.get("max_test_samples")

        method_cap = self._expl_cfg.get("method_max_instances")
        if method_cap is not None:
            try:
                cap_value = int(method_cap)
            except (TypeError, ValueError):
                raise ValueError(
                    f"experiment.explanation.method_max_instances must be an integer. Got {method_cap!r}"
                ) from None
            if cap_value <= 0:
                raise ValueError(
                    f"experiment.explanation.method_max_instances must be > 0. Got {cap_value}"
                )
            if max_n is None:
                max_n = cap_value
            else:
                max_n = min(int(max_n), cap_value)

        original_len = len(X)
        self._sampling_info.setdefault("strategy", "sequential")
        self._sampling_info["original_size"] = original_len
        self._sampling_info["selected_size"] = original_len
        self._sampling_info["max_instances"] = int(max_n) if max_n is not None else None
        base_indices = np.arange(original_len, dtype=int)

        if max_n is None or original_len <= int(max_n):
            self._sample_indices = base_indices
            return X, y

        problem_type = self._infer_problem_type(y)
        self._sampling_info["problem_type"] = problem_type

        strategy_cfg = str(self._expl_cfg.get("sampling_strategy", "sequential")).lower()
        if strategy_cfg == "auto":
            if problem_type == "classification":
                strategy = "balanced"
            elif problem_type == "regression":
                strategy = "quantile"
            else:
                strategy = "random"
        else:
            strategy = strategy_cfg
        self._sampling_info["strategy"] = strategy

        indices = self._select_sample_indices(
            n_samples=original_len,
            y=y,
            max_n=int(max_n),
            strategy=strategy,
            problem_type=problem_type,
        )
        if len(indices) != int(max_n):
            indices = np.arange(int(max_n))

        X = X[indices]
        if y is not None:
            y = y[indices]
        self._sampling_info["selected_size"] = len(X)
        self._sample_indices = indices
        return X, y

    def _coerce_X_y(
        self, X: ArrayLike, y: Optional[ArrayLike]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_np = self._to_numpy_2d(X)
        y_np = None
        if y is not None:
            y_np = self._to_numpy_1d(y)
        return X_np, y_np

    def _to_numpy_2d(self, X: ArrayLike) -> np.ndarray:
        if _HAS_PANDAS and isinstance(X, pd.DataFrame):
            return X.values
        if isinstance(X, (list, tuple)):
            X = np.asarray(X)
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X.reshape(1, -1)
            if X.ndim == 2:
                return X
        raise TypeError("X must be a 1D/2D array-like (list, np.ndarray, or pandas DataFrame).")

    def _to_numpy_1d(self, y: ArrayLike) -> np.ndarray:
        if _HAS_PANDAS and isinstance(y, pd.Series):
            return y.values
        if isinstance(y, (list, tuple)):
            y = np.asarray(y)
        if isinstance(y, np.ndarray):
            if y.ndim == 0:
                return y.reshape(1)
            if y.ndim == 1:
                return y
            if y.ndim == 2 and y.shape[1] == 1:
                return y.ravel()
        raise TypeError("y must be a 1D array-like (list, np.ndarray, or pandas Series).")

    def _row_to_instance(self, X: ArrayLike, idx: int) -> InstanceLike:
        if _HAS_PANDAS and isinstance(X, pd.DataFrame):
            return X.iloc[idx]
        X_np = self._to_numpy_2d(X)
        return X_np[idx, :]

    def _predict(self, X: ArrayLike) -> np.ndarray:
        X_np = self._to_numpy_2d(X)
        preds = self.model.predict(X_np)
        return np.asarray(preds)

    def _predict_numeric(self, X: ArrayLike) -> np.ndarray:
        X_np = self._to_numpy_2d(X)
        predict_numeric = getattr(self.model, "predict_numeric", None)
        if callable(predict_numeric):
            preds = predict_numeric(X_np)
        else:
            preds = self.model.predict(X_np)
        arr = np.asarray(preds)
        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(float)
        try:
            return arr.astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Model predictions are not numeric; implement predict_numeric to continue"
            ) from exc

    def _predict_proba(self, X: ArrayLike) -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            X_np = self._to_numpy_2d(X)
            try:
                proba = self.model.predict_proba(X_np)
                return np.asarray(proba)
            except Exception as exc:
                self.logger.debug("predict_proba failed: %s", exc)
                return None
        return None

    def _timeit(self, fn, *args, **kwargs) -> Tuple[Any, float]:
        t0 = time.time()
        out = fn(*args, **kwargs)
        return out, time.time() - t0

    def _standardize_explanation_output(
        self,
        *,
        attributions: Union[np.ndarray, List[float]],
        instance: InstanceLike,
        prediction: Any,
        prediction_proba: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        per_instance_time: float = 0.0,
    ) -> Dict[str, Any]:
        if feature_names is None:
            feature_names = self._infer_feature_names(instance)

        return {
            "method": self.config.get("type", self.__class__.__name__),
            "prediction": prediction,
            "prediction_proba": prediction_proba,
            "attributions": attributions,
            "feature_names": feature_names,
            "metadata": metadata or {},
            "generation_time": per_instance_time,
            "instance": np.asarray(instance).tolist(),
        }

    def _infer_feature_names(self, instance: InstanceLike) -> List[str]:
        names = getattr(self.dataset, "feature_names", None)
        if names is not None:
            return list(names)

        if _HAS_PANDAS and hasattr(instance, "index"):
            try:
                return list(instance.index)
            except Exception:
                pass

        instance_array = np.asarray(instance)
        if instance_array.ndim == 0:
            return ["feature_0"]
        if instance_array.ndim == 1:
            return [f"feature_{i}" for i in range(instance_array.shape[0])]
        return [f"feature_{i}" for i in range(instance_array.shape[-1])]

    def _infer_problem_type(self, y: Optional[np.ndarray]) -> str:
        cfg_override = self._expl_cfg.get("problem_type")
        if isinstance(cfg_override, str):
            override = cfg_override.lower()
            if override in {"classification", "regression"}:
                return override

        dataset_task = getattr(self.dataset, "task", None) or getattr(
            self.dataset, "task_type", None
        )
        if isinstance(dataset_task, str):
            lowered = dataset_task.lower()
            if lowered in {"classification", "regression"}:
                return lowered

        estimator_type = getattr(self.model, "_estimator_type", None)
        if estimator_type in {"classifier", "regressor"}:
            return "classification" if estimator_type == "classifier" else "regression"

        if y is None:
            return "unknown"

        y_arr = np.asarray(y)
        if y_arr.dtype.kind in {"U", "S", "O"}:
            return "classification"
        unique_vals = np.unique(y_arr)
        if y_arr.dtype.kind in {"b", "i", "u"}:
            if len(unique_vals) <= max(15, int(0.1 * len(y_arr))):
                return "classification"
        return "regression"

    def _select_sample_indices(
        self,
        *,
        n_samples: int,
        y: Optional[np.ndarray],
        max_n: int,
        strategy: str,
        problem_type: str,
    ) -> np.ndarray:
        def _finalize(selected: np.ndarray) -> np.ndarray:
            selected = np.asarray(selected, dtype=int)
            return self._ensure_min_class_labels(
                selected,
                y=y,
                problem_type=problem_type,
                min_classes=2,
            )

        if max_n >= n_samples:
            return _finalize(np.arange(n_samples, dtype=int))

        stratified_aliases = {"balanced", "stratified", "class_balanced"}
        quantile_aliases = {"quantile", "diverse", "range"}

        if strategy in stratified_aliases and y is not None and problem_type == "classification":
            return _finalize(self._balanced_class_indices(y, max_n))

        if strategy in quantile_aliases and y is not None and problem_type == "regression":
            return _finalize(self._quantile_sample_indices(y, max_n))

        if strategy in {"random", "shuffle"}:
            rng = np.random.default_rng(self.random_state)
            return _finalize(rng.choice(n_samples, size=max_n, replace=False))

        return _finalize(np.arange(max_n, dtype=int))

    def _balanced_class_indices(self, y: np.ndarray, max_n: int) -> np.ndarray:
        unique = np.unique(y)
        if len(unique) <= 1:
            return np.arange(min(len(y), max_n), dtype=int)

        rng = np.random.default_rng(self.random_state)
        buckets = {label: rng.permutation(np.where(y == label)[0]) for label in unique}
        selected: list[int] = []
        while len(selected) < max_n and any(len(indices) for indices in buckets.values()):
            for label in unique:
                indices = buckets[label]
                if len(indices) == 0 or len(selected) >= max_n:
                    continue
                selected.append(int(indices[0]))
                buckets[label] = indices[1:]
        return np.asarray(selected[:max_n], dtype=int)

    def _quantile_sample_indices(self, y: np.ndarray, max_n: int) -> np.ndarray:
        if len(y) <= max_n:
            return np.arange(len(y), dtype=int)
        quantiles = np.linspace(0.0, 1.0, num=max_n, endpoint=False)
        order = np.argsort(y, kind="mergesort")
        return np.unique(order[np.floor(quantiles * len(order)).astype(int)])[:max_n]

    def _ensure_min_class_labels(
        self,
        selected: np.ndarray,
        *,
        y: Optional[np.ndarray],
        problem_type: str,
        min_classes: int,
    ) -> np.ndarray:
        if y is None or problem_type != "classification" or selected.size == 0:
            return np.asarray(selected, dtype=int)

        selected = np.asarray(selected, dtype=int)
        present = np.unique(y[selected])
        if len(present) >= min_classes:
            return selected

        needed = [label for label in np.unique(y) if label not in present]
        if not needed:
            return selected

        extras: list[int] = []
        for label in needed:
            matches = np.where(y == label)[0]
            if matches.size:
                extras.append(int(matches[0]))

        merged = np.unique(np.concatenate([selected, np.asarray(extras, dtype=int)]))
        return merged

    def _augment_metadata(
        self,
        explanations: List[Dict[str, Any]],
        y: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        if y is None:
            return explanations

        for idx, explanation in enumerate(explanations):
            metadata = explanation.setdefault("metadata", {})
            metadata.setdefault("target", np.asarray(y[idx]).item())
        return explanations
