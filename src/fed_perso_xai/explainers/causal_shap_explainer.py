"""
Causal SHAP explainer for tabular data.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._background_data import require_client_local_background, sample_client_local_background
from .base import ArrayLike, BaseExplainer, InstanceLike


_MIN_CORRELATION_SAMPLE_ROWS = 2


class CausalSHAPExplainer(BaseExplainer):
    """
    Approximate SHAP values while respecting a simple causal ordering inferred
    from feature correlations over the sampled client-local background data.
    Very small local samples can make the inferred graph unstable, so graph
    inference warnings and diagnostics are exposed in metadata.
    """

    supported_data_types = ["tabular"]
    supported_model_types = [
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "generic-predict",
    ]

    def __init__(self, config: Dict[str, Any], model: Any, dataset: Any):
        super().__init__(config=config, model=model, dataset=dataset)
        self._X_train: Optional[np.ndarray] = getattr(dataset, "X_train", None)
        self._y_train: Optional[np.ndarray] = getattr(dataset, "y_train", None)
        self._rng = np.random.default_rng(self.random_state)
        self._causal_graph_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self._baseline_vector: Optional[np.ndarray] = None
        self._baseline_prediction: Optional[float] = None
        self._background: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        X_np, y_np = self._coerce_X_y(X, y)
        require_client_local_background(self._expl_cfg)
        self._graph_inference_warning_threshold()
        self._background = sample_client_local_background(
            X_np,
            expl_cfg=self._expl_cfg,
            random_state=self.random_state,
        )
        self._X_train = self._background
        self._y_train = y_np
        self._baseline_vector = None
        self._baseline_prediction = None
        self._causal_graph_cache.clear()

    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        inst2d = self._to_numpy_2d(instance)
        inst_vec = inst2d[0]

        X_train = self._ensure_training_data(inst_vec)
        feature_names = self._infer_feature_names(inst_vec)

        causal_graph, graph_info = self._infer_causal_structure(X_train, feature_names)
        attributions, info = self._causal_shap(inst_vec, X_train, causal_graph, feature_names)

        prediction, t_pred = self._timeit(self._predict_numeric, inst2d)
        prediction_proba = self._predict_proba(inst2d)

        pred_arr = np.asarray(prediction).ravel()
        pred_value = float(pred_arr[0]) if pred_arr.size else float(pred_arr)

        proba_value = None
        if prediction_proba is not None:
            proba_value = np.asarray(prediction_proba)[0]

        metadata = {
            "causal_graph": causal_graph,
            "coalition_samples": info["coalition_samples"],
            "correlation_threshold": info["correlation_threshold"],
            "baseline_source": info["baseline_source"],
            "baseline_instance": info["baseline_instance"],
            "baseline_prediction": info["baseline_prediction"],
            "background_data_source": self._expl_cfg.get(
                "background_data_source",
                "client_local_train",
            ),
            "background_sample_size": (
                None if self._background is None else int(self._background.shape[0])
            ),
            "graph_inference_sample_size": graph_info["graph_inference_sample_size"],
            "graph_inference_small_sample_warning": graph_info[
                "graph_inference_small_sample_warning"
            ],
            "graph_inference_small_sample_threshold": graph_info[
                "graph_inference_small_sample_threshold"
            ],
            "graph_inference_fallback": graph_info["graph_inference_fallback"],
            "graph_inference_nan_count": graph_info["graph_inference_nan_count"],
            "graph_inference_warning_messages": list(
                graph_info["graph_inference_warning_messages"]
            ),
        }
        if proba_value is not None:
            metadata["explained_class"] = int(np.argmax(np.asarray(proba_value, dtype=float)))

        return self._standardize_explanation_output(
            attributions=attributions.tolist(),
            instance=inst_vec,
            prediction=pred_value,
            prediction_proba=proba_value,
            feature_names=feature_names,
            metadata=metadata,
            per_instance_time=t_pred,
        )

    def explain_batch(self, X: ArrayLike) -> List[Dict[str, Any]]:
        """
        Batch wrapper that reuses shared predictions while each instance still
        runs its causal-SHAP sampling.
        """
        X_np, _ = self._coerce_X_y(X, None)

        if len(X_np) == 0:
            return []

        batch_start = time.time()
        preds = np.asarray(self._predict_numeric(X_np))
        proba = self._predict_proba(X_np)

        results: List[Dict[str, Any]] = []
        for idx, inst_vec in enumerate(X_np):
            X_train = self._ensure_training_data(inst_vec)
            feature_names = self._infer_feature_names(inst_vec)
            causal_graph, graph_info = self._infer_causal_structure(X_train, feature_names)
            attributions, info = self._causal_shap(
                inst_vec, X_train, causal_graph, feature_names
            )

            pred_row = np.asarray(preds[idx]).ravel()
            pred_value = float(pred_row[0]) if pred_row.size else float(pred_row)

            proba_value = None
            if proba is not None:
                proba_value = np.asarray(proba[idx])

            metadata = {
                "causal_graph": causal_graph,
                "coalition_samples": info["coalition_samples"],
                "correlation_threshold": info["correlation_threshold"],
                "baseline_source": info["baseline_source"],
                "baseline_instance": info["baseline_instance"],
                "baseline_prediction": info["baseline_prediction"],
                "background_data_source": self._expl_cfg.get(
                    "background_data_source",
                    "client_local_train",
                ),
                "background_sample_size": (
                    None if self._background is None else int(self._background.shape[0])
                ),
                "graph_inference_sample_size": graph_info["graph_inference_sample_size"],
                "graph_inference_small_sample_warning": graph_info[
                    "graph_inference_small_sample_warning"
                ],
                "graph_inference_small_sample_threshold": graph_info[
                    "graph_inference_small_sample_threshold"
                ],
                "graph_inference_fallback": graph_info["graph_inference_fallback"],
                "graph_inference_nan_count": graph_info["graph_inference_nan_count"],
                "graph_inference_warning_messages": list(
                    graph_info["graph_inference_warning_messages"]
                ),
            }
            if proba_value is not None:
                metadata["explained_class"] = int(
                    np.argmax(np.asarray(proba_value, dtype=float))
                )

            results.append(
                self._standardize_explanation_output(
                    attributions=attributions.tolist(),
                    instance=inst_vec,
                    prediction=pred_value,
                    prediction_proba=proba_value,
                    feature_names=feature_names,
                    metadata=metadata,
                    per_instance_time=0.0,
                )
            )
        total_time = time.time() - batch_start
        avg_time = total_time / len(results) if results else 0.0
        for record in results:
            record["generation_time"] = avg_time
        return results

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "background_data_source": self._expl_cfg.get(
                    "background_data_source",
                    "client_local_train",
                ),
                "background_sample_size": (
                    None if self._background is None else int(self._background.shape[0])
                ),
                "graph_inference_small_sample_threshold": (
                    self._graph_inference_warning_threshold()
                ),
            }
        )
        return info

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_training_data(self, fallback_instance: np.ndarray) -> np.ndarray:
        if self._X_train is not None:
            return self._X_train
        self._X_train = fallback_instance.reshape(1, -1)
        return self._X_train

    def _graph_inference_warning_threshold(self) -> int:
        """Return the explicit heuristic threshold for graph-size warnings."""

        threshold = int(self._expl_cfg.get("causal_shap_min_graph_samples_warning", 10))
        if threshold < _MIN_CORRELATION_SAMPLE_ROWS:
            raise ValueError(
                "experiment.explanation.causal_shap_min_graph_samples_warning must be "
                f">= {_MIN_CORRELATION_SAMPLE_ROWS}. Got {threshold}."
            )
        return threshold

    def _warn_graph_inference(self, message: str) -> None:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        self.logger.warning(message)

    def _infer_causal_structure(
        self, X_train: np.ndarray, feature_names: List[str]
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        key = tuple(feature_names)
        cached = self._causal_graph_cache.get(key)
        if cached is not None:
            return cached["graph"], cached["info"]

        n_features = len(feature_names)
        n_samples = int(X_train.shape[0]) if X_train.ndim == 2 else 0
        small_sample_threshold = self._graph_inference_warning_threshold()
        small_sample_warning = n_samples < small_sample_threshold
        corr_threshold = float(self._expl_cfg.get("causal_shap_corr_threshold", 0.3))
        graph: Dict[str, List[str]] = {fname: [] for fname in feature_names}
        warning_messages: List[str] = []

        def record_warning(message: str) -> None:
            warning_messages.append(message)
            self._warn_graph_inference(message)

        if small_sample_warning:
            record_warning(
                "Causal SHAP graph inference is using a small client-local background "
                f"sample (n_samples={n_samples}, threshold={small_sample_threshold}); "
                "inferred feature correlations may be unstable and should be interpreted cautiously."
            )

        info: Dict[str, Any] = {
            "graph_inference_sample_size": n_samples,
            "graph_inference_small_sample_warning": small_sample_warning,
            "graph_inference_small_sample_threshold": small_sample_threshold,
            "graph_inference_fallback": None,
            "graph_inference_nan_count": 0,
            "graph_inference_warning_messages": warning_messages,
        }

        if n_features == 0:
            self._causal_graph_cache[key] = {"graph": graph, "info": info}
            return graph, info

        if n_samples < _MIN_CORRELATION_SAMPLE_ROWS:
            info["graph_inference_fallback"] = "empty_graph_insufficient_rows"
            record_warning(
                "Causal SHAP graph inference could not compute correlations from the "
                f"client-local background sample because only {n_samples} row(s) were available; "
                "falling back to an empty causal graph."
            )
            self._causal_graph_cache[key] = {"graph": graph, "info": info}
            return graph, info

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(np.asarray(X_train, dtype=np.float64), rowvar=False)
        corr = np.asarray(corr, dtype=np.float64)

        if corr.ndim == 0:
            corr = corr.reshape(1, 1)

        expected_shape = (n_features, n_features)
        if corr.shape != expected_shape:
            info["graph_inference_fallback"] = "empty_graph_invalid_correlation_shape"
            record_warning(
                "Causal SHAP graph inference produced an unexpected correlation matrix "
                f"shape {corr.shape} for {n_features} feature(s); falling back to an empty causal graph."
            )
            self._causal_graph_cache[key] = {"graph": graph, "info": info}
            return graph, info

        nan_count = int(np.isnan(corr).sum())
        if nan_count > 0:
            info["graph_inference_nan_count"] = nan_count
            record_warning(
                "Causal SHAP graph inference encountered NaN correlations in the "
                f"client-local background sample (nan_count={nan_count}); replacing them with 0.0 before building the graph."
            )
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        for i, fname in enumerate(feature_names):
            parents: List[str] = []
            for j in range(len(feature_names)):
                if i == j:
                    continue
                if abs(corr[i, j]) >= corr_threshold and j < i:
                    parents.append(feature_names[j])
            graph[fname] = parents
        self._causal_graph_cache[key] = {"graph": graph, "info": info}
        return graph, info

    def _causal_shap(
        self,
        instance: np.ndarray,
        X_train: np.ndarray,
        causal_graph: Dict[str, List[str]],
        feature_names: List[str],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        baseline = self._baseline_vector
        baseline_pred = self._baseline_prediction
        if baseline is None:
            baseline = np.mean(X_train, axis=0)
            self._baseline_vector = baseline
        if baseline_pred is None:
            baseline_pred = float(
                np.asarray(self._predict_numeric(baseline.reshape(1, -1))).ravel()[0]
            )
            self._baseline_prediction = baseline_pred
        base_pred = float(np.asarray(self._predict_numeric(instance.reshape(1, -1))).ravel()[0])

        n_features = len(instance)
        coalition_samples = int(self._expl_cfg.get("causal_shap_coalitions", 50))

        contributions = np.zeros(n_features)
        for idx, fname in enumerate(feature_names):
            parents = causal_graph.get(fname, [])
            parent_indices = [feature_names.index(p) for p in parents if p in feature_names]
            marginal_effects = []

            for _ in range(coalition_samples):
                coalition = self._sample_coalition(idx, n_features, parent_indices)
                inst_without = baseline.copy()
                inst_without[coalition] = instance[coalition]
                pred_without = float(
                    np.asarray(self._predict_numeric(inst_without.reshape(1, -1))).ravel()[0]
                )

                inst_with = inst_without.copy()
                inst_with[idx] = instance[idx]
                pred_with = float(
                    np.asarray(self._predict_numeric(inst_with.reshape(1, -1))).ravel()[0]
                )

                marginal_effects.append(pred_with - pred_without)

            contributions[idx] = np.mean(marginal_effects)

        current_sum = contributions.sum()
        total_effect = base_pred - baseline_pred
        if abs(current_sum) > 1e-12:
            contributions *= total_effect / current_sum

        info = {
            "coalition_samples": coalition_samples,
            "correlation_threshold": float(
                self._expl_cfg.get("causal_shap_corr_threshold", 0.3)
            ),
            "baseline_source": self._baseline_source(),
            "baseline_instance": np.asarray(baseline, dtype=float).reshape(-1).tolist(),
            "baseline_prediction": float(baseline_pred),
        }
        return contributions, info

    def _baseline_source(self) -> str:
        if self._background is not None:
            return "background_mean"
        if self._X_train is not None and getattr(self._X_train, "shape", (0,))[0] > 1:
            return "train_mean"
        return "instance_fallback"

    def _sample_coalition(
        self,
        feature_idx: int,
        n_features: int,
        parent_indices: List[int],
    ) -> List[int]:
        coalition_size = self._rng.integers(low=0, high=max(1, n_features - 1))
        coalition = []
        for parent in parent_indices:
            if self._rng.random() < 0.8:
                coalition.append(parent)

        others = [i for i in range(n_features) if i != feature_idx and i not in coalition]
        self._rng.shuffle(others)
        coalition.extend(others[: max(0, coalition_size - len(coalition))])
        return coalition
