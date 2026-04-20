"""
Integrated Gradients explainer for tabular data.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._background_data import build_client_local_mean_reference, require_client_local_background
from .base import ArrayLike, BaseExplainer, InstanceLike


class IntegratedGradientsExplainer(BaseExplainer):
    """Finite-difference Integrated Gradients for models with smooth outputs."""

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
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._train_mean: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None
        self._background_data_source = str(
            self._expl_cfg.get("background_data_source", "client_local_train")
        )

        ds_X = getattr(self.dataset, "X_train", None)
        ds_y = getattr(self.dataset, "y_train", None)
        if ds_X is not None:
            self._cache_training_stats(ds_X, ds_y)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Cache training arrays/statistics for later perturbations."""
        require_client_local_background(self._expl_cfg)
        self._cache_training_stats(X, y)

    def is_compatible(self) -> bool:
        """Require either regression outputs or probabilities for smoother gradients."""
        if not super().is_compatible():
            return False

        estimator_type = getattr(self.model, "_estimator_type", None)
        has_proba = hasattr(self.model, "predict_proba")
        allow_non_diff = bool(self._expl_cfg.get("ig_allow_nondifferentiable", False))

        if estimator_type == "regressor" or has_proba:
            return True

        if not allow_non_diff:
            self.logger.warning(
                "Integrated Gradients needs a differentiable/predict_proba model. "
                "Set experiment.explanation.ig_allow_nondifferentiable=True to override."
            )
            return False
        return True

    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        inst2d = self._to_numpy_2d(instance)
        inst_vec = inst2d[0]
        self._ensure_training_cache(inst_vec)

        (attributions, info), t_attr = self._timeit(self._integrated_gradients, inst_vec)
        prediction, t_pred = self._timeit(self._predict_numeric, inst2d)
        prediction_proba = self._predict_proba(inst2d)

        metadata = {
            "baseline_source": info["baseline_source"],
            "n_steps": info["n_steps"],
            "epsilon": info["epsilon"],
        }

        pred_arr = np.asarray(prediction).ravel()
        pred_value = float(pred_arr[0]) if pred_arr.size else float(pred_arr)

        proba_value = None
        if prediction_proba is not None:
            proba_value = np.asarray(prediction_proba)[0]

        return self._standardize_explanation_output(
            attributions=attributions.tolist(),
            instance=inst_vec,
            prediction=pred_value,
            prediction_proba=proba_value,
            metadata=metadata,
            per_instance_time=t_attr + t_pred,
        )

    def explain_batch(self, X: ArrayLike) -> List[Dict[str, Any]]:
        """
        Vectorized entry point so we only coerce data/predictions once while
        still computing integrated gradients per instance.
        """
        X_np, _ = self._coerce_X_y(X, None)

        if len(X_np) == 0:
            return []

        batch_start = time.time()
        preds = np.asarray(self._predict_numeric(X_np))
        proba = self._predict_proba(X_np)

        results: List[Dict[str, Any]] = []
        for idx, inst_vec in enumerate(X_np):
            self._ensure_training_cache(inst_vec)
            attributions, info = self._integrated_gradients(inst_vec)

            pred_row = np.asarray(preds[idx]).ravel()
            pred_value = float(pred_row[0]) if pred_row.size else float(pred_row)

            proba_value = None
            if proba is not None:
                proba_value = np.asarray(proba[idx])

            metadata = {
                "baseline_source": info["baseline_source"],
                "n_steps": info["n_steps"],
                "epsilon": info["epsilon"],
            }

            results.append(
                self._standardize_explanation_output(
                    attributions=attributions.tolist(),
                    instance=inst_vec,
                    prediction=pred_value,
                    prediction_proba=proba_value,
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
                "background_data_source": self._background_data_source,
                "baseline_source": "train_mean" if self._train_mean is not None else "zeros",
            }
        )
        return info

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _cache_training_stats(
        self, X: ArrayLike, y: Optional[ArrayLike]
    ) -> None:
        if X is None:
            return
        X_np, y_np = self._coerce_X_y(X, y)
        self._X_train = X_np
        self._y_train = y_np
        if X_np.shape[0] > 0:
            self._train_mean = build_client_local_mean_reference(
                X_np,
                expl_cfg=self._expl_cfg,
            )
        else:
            self._train_mean = None
        std = np.std(X_np, axis=0)
        std[std == 0.0] = 1e-6
        self._train_std = std

    def _ensure_training_cache(self, fallback_instance: np.ndarray) -> None:
        if self._X_train is not None:
            return
        self._cache_training_stats(fallback_instance.reshape(1, -1), None)

    def _integrated_gradients(
        self, instance: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        baseline, baseline_source = self._baseline(instance)
        diff = instance - baseline

        n_steps = max(10, int(self._expl_cfg.get("ig_steps", 100)))
        epsilon = float(self._expl_cfg.get("ig_epsilon", 1e-4))

        alphas = np.linspace(0.0, 1.0, n_steps, endpoint=True)
        gradients = np.zeros((n_steps, len(instance)))

        for i, alpha in enumerate(alphas):
            point = baseline + alpha * diff
            gradients[i] = self._finite_difference_gradient(point, epsilon)

        avg_grad = gradients.mean(axis=0)
        attributions = diff * avg_grad

        info = {
            "baseline_source": baseline_source,
            "n_steps": n_steps,
            "epsilon": epsilon,
        }
        return attributions, info

    def _baseline(self, instance: np.ndarray) -> Tuple[np.ndarray, str]:
        if self._train_mean is not None:
            return self._train_mean, "train_mean"
        zeros = np.zeros_like(instance)
        return zeros, "zeros"

    def _finite_difference_gradient(
        self, point: np.ndarray, epsilon: float
    ) -> np.ndarray:
        n_features = len(point)
        if n_features == 0:
            return np.zeros_like(point, dtype=float)
        perturbations = np.repeat(point.reshape(1, -1), 2 * n_features, axis=0)
        indices = np.arange(n_features)
        perturbations[:n_features, indices] += epsilon
        perturbations[n_features:, indices] -= epsilon
        scores = self._batched_scalar_predictions(perturbations)
        plus_vals = scores[:n_features]
        minus_vals = scores[n_features:]
        grad = (plus_vals - minus_vals) / (2 * epsilon)
        return grad

    def _scalar_prediction(self, point: np.ndarray) -> float:
        row = point.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(row)).ravel()
            target = self._expl_cfg.get("ig_target_class")
            if target is not None:
                target_idx = int(target)
                if 0 <= target_idx < len(proba):
                    return float(proba[target_idx])
            if len(proba) > 1:
                return float(proba[1])
            return float(proba[0])

        preds = np.asarray(self._predict_numeric(row)).ravel()
        return float(preds[0])

    def _batched_scalar_predictions(self, points: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(points))
            target = self._expl_cfg.get("ig_target_class")
            if target is not None:
                target_idx = int(target)
                if proba.ndim > 1 and 0 <= target_idx < proba.shape[1]:
                    return proba[:, target_idx].astype(float)
            if proba.ndim > 1:
                if proba.shape[1] > 1:
                    return proba[:, 1].astype(float)
                return proba[:, 0].astype(float)
            return proba.astype(float).ravel()

        preds = np.asarray(self._predict_numeric(points), dtype=float)
        return preds.ravel()
