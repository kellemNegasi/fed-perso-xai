"""
LIME explainer specialized for local explanations on tabular data.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from ._background_data import build_client_local_mean_reference, require_client_local_background
from .base import ArrayLike, BaseExplainer, InstanceLike


class LIMEExplainer(BaseExplainer):
    """Local Interpretable Model-agnostic Explanations for tabular data."""

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
        self._rng = np.random.default_rng(self.random_state)

        ds_X = getattr(self.dataset, "X_train", None)
        ds_y = getattr(self.dataset, "y_train", None)
        if ds_X is not None:
            self._cache_training_stats(ds_X, ds_y)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Cache training arrays/statistics for later perturbations."""
        require_client_local_background(self._expl_cfg)
        self._cache_training_stats(X, y)

    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        """Generate a local linear surrogate around a single instance."""
        inst2d = self._to_numpy_2d(instance)
        inst_vec = inst2d[0]
        self._ensure_training_cache(inst_vec)

        prediction, t_pred = self._timeit(self._predict_numeric, inst2d)
        prediction_proba = self._predict_proba(inst2d)
        proba_value = None
        if prediction_proba is not None:
            proba_value = np.asarray(prediction_proba)[0]
        target_class = self._resolve_target_class(proba_value)
        (attributions, info), t_lime = self._timeit(
            self._generate_local_explanation,
            inst_vec,
            target_class,
        )

        pred_arr = np.asarray(prediction).ravel()
        pred_value = float(pred_arr[0]) if pred_arr.size else float(pred_arr)

        metadata = self._build_metadata(
            info=info,
            target_class=target_class,
        )
        return self._standardize_explanation_output(
            attributions=attributions.tolist(),
            instance=inst_vec,
            prediction=pred_value,
            prediction_proba=proba_value,
            metadata=metadata,
            per_instance_time=t_lime + t_pred,
        )

    def explain_batch(self, X: ArrayLike) -> List[Dict[str, Any]]:
        """
        Run LIME over a batch while sharing coercion/prediction work and still
        generating per-instance local surrogate fits.
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

            pred_row = np.asarray(preds[idx]).ravel()
            pred_value = float(pred_row[0]) if pred_row.size else float(pred_row)

            proba_value = None
            if proba is not None:
                proba_value = np.asarray(proba[idx])
            target_class = self._resolve_target_class(proba_value)
            attributions, info = self._generate_local_explanation(inst_vec, target_class)

            results.append(
                self._standardize_explanation_output(
                    attributions=attributions.tolist(),
                    instance=inst_vec,
                    prediction=pred_value,
                    prediction_proba=proba_value,
                    metadata=self._build_metadata(
                        info=info,
                        target_class=target_class,
                    ),
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

    def _cache_training_stats(self, X: ArrayLike, y: Optional[ArrayLike]) -> None:
        """Store training arrays plus mean/std statistics."""
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
        """Guarantee that perturbation stats exist, even if fit() was skipped."""
        if self._X_train is not None:
            return
        self._cache_training_stats(fallback_instance.reshape(1, -1), None)

    def _generate_local_explanation(
        self,
        instance: np.ndarray,
        target_class: Optional[int],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perturb the instance, fit a weighted linear model, and return coefficients."""
        n_features = instance.shape[0]
        n_samples = int(self._expl_cfg.get("lime_num_samples", 100))
        kernel_width = float(
            self._expl_cfg.get("lime_kernel_width", np.sqrt(n_features) * 0.75)
        )
        noise_scale = float(self._expl_cfg.get("lime_noise_scale", 0.1))
        alpha = float(self._expl_cfg.get("lime_alpha", 1e-2))

        std = self._train_std if self._train_std is not None else np.ones_like(instance)
        perturbations = instance + self._rng.normal(
            0.0,
            std * noise_scale,
            size=(n_samples, n_features),
        )
        perturbations = np.vstack([instance, perturbations])

        preds = np.asarray(self._predict_numeric(perturbations))
        target = self._local_target_vector(
            perturbations,
            preds,
            target_class=target_class,
        )

        distances = np.linalg.norm(perturbations - instance, axis=1)
        weights = np.exp(-(distances**2) / (kernel_width**2 + 1e-12))
        weights[0] = weights.max()

        linear_model = Ridge(alpha=alpha)
        linear_model.fit(perturbations, target, sample_weight=weights)
        importance = np.abs(linear_model.coef_)

        info = {
            "num_samples": n_samples,
            "kernel_width": kernel_width,
            "noise_scale": noise_scale,
            "alpha": alpha,
            "target_class": target_class,
        }
        return importance, info

    def _build_metadata(
        self,
        *,
        info: Dict[str, Any],
        target_class: Optional[int],
    ) -> Dict[str, Any]:
        baseline_prediction = None
        baseline_instance = None
        if self._train_mean is not None:
            baseline_instance = np.asarray(self._train_mean, dtype=float).reshape(-1).tolist()
            baseline_prediction = float(
                np.asarray(self._predict_numeric(self._train_mean.reshape(1, -1))).ravel()[0]
            )

        metadata: Dict[str, Any] = {
            "baseline_source": "train_mean" if baseline_instance is not None else "zeros",
            "baseline_instance": baseline_instance,
            "baseline_prediction": baseline_prediction,
            "background_data_source": self._background_data_source,
            "num_samples": info["num_samples"],
            "kernel_width": info["kernel_width"],
            "noise_scale": info["noise_scale"],
            "alpha": info["alpha"],
        }
        if target_class is not None:
            metadata["explained_class"] = int(target_class)
        return metadata

    def _local_target_vector(
        self,
        perturbations: np.ndarray,
        predictions: np.ndarray,
        *,
        target_class: Optional[int],
    ) -> np.ndarray:
        """Return numeric targets for the local surrogate regression."""
        proba = self._predict_proba(perturbations)
        if proba is not None:
            proba_arr = np.asarray(proba)
            if proba_arr.ndim == 1:
                return proba_arr.reshape(-1)
            if proba_arr.ndim == 2:
                if proba_arr.shape[1] == 1:
                    return proba_arr.ravel()
                if target_class is None:
                    raise ValueError(
                        "LIME target_class must be resolved from config or the original instance "
                        "before building surrogate targets."
                    )
                if not 0 <= int(target_class) < proba_arr.shape[1]:
                    raise ValueError(
                        f"LIME target_class={target_class} is out of bounds for "
                        f"{proba_arr.shape[1]} probability columns."
                    )
                return proba_arr[:, int(target_class)]
            flat = proba_arr.reshape(proba_arr.shape[0], -1)
            return flat[:, 0]
        return self._encode_prediction_labels(predictions)

    def _resolve_target_class(self, prediction_proba: Optional[np.ndarray]) -> Optional[int]:
        """
        Resolve the single class index explained by one LIME surrogate.

        Binary classification keeps the repository's existing positive-class
        convention (class index 1) unless an explicit ``lime_target_class`` is
        configured. Multiclass defaults to the original instance's predicted
        class, and that same fixed class is used for every perturbation.
        """
        explicit_target = self._expl_cfg.get("lime_target_class")
        if explicit_target is not None:
            target_class = int(explicit_target)
            class_count = self._class_count(prediction_proba)
            if class_count is not None and not 0 <= target_class < class_count:
                raise ValueError(
                    f"experiment.explanation.lime_target_class={target_class} is out of bounds "
                    f"for {class_count} classes."
                )
            return target_class

        class_count = self._class_count(prediction_proba)
        if class_count == 2:
            return 1

        proba_row = self._probability_row(prediction_proba)
        if proba_row is not None and proba_row.size > 0:
            return int(np.argmax(proba_row))
        return None

    def _class_count(self, prediction_proba: Optional[np.ndarray]) -> Optional[int]:
        proba_row = self._probability_row(prediction_proba)
        if proba_row is not None and proba_row.size > 1:
            return int(proba_row.size)

        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return None
        try:
            return int(len(classes))
        except TypeError:
            return None

    def _probability_row(self, prediction_proba: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if prediction_proba is None:
            return None
        proba_arr = np.asarray(prediction_proba, dtype=float)
        if proba_arr.size == 0:
            return None
        if proba_arr.ndim == 0:
            return proba_arr.reshape(1)
        if proba_arr.ndim == 1:
            return proba_arr.reshape(-1)
        return np.asarray(proba_arr[0], dtype=float).reshape(-1)

    def _encode_prediction_labels(self, predictions: np.ndarray) -> np.ndarray:
        preds = np.asarray(predictions)
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.ravel()
        if np.issubdtype(preds.dtype, np.number):
            return preds.astype(float)
        indices = self._prediction_indices(predictions)
        return indices.astype(float)
