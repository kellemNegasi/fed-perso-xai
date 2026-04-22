"""
SHAP explainer for local (per-instance) explanations on tabular data only.
Depends on BaseExplainer from fed_perso_xai.explainers.base.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from ._background_data import require_client_local_background, sample_client_local_background
from .base import ArrayLike, BaseExplainer, InstanceLike


class SHAPExplainer(BaseExplainer):
    """Minimal SHAP explainer for tabular data."""

    supported_data_types = ["tabular"]
    supported_model_types = ["sklearn", "xgboost", "lightgbm", "catboost", "generic-predict"]

    def __init__(self, config: Dict[str, Any], model: Any, dataset: Any):
        super().__init__(config, model, dataset)
        self._shap = None
        self._explainer = None
        self._is_tree = False
        self._explainer_type: Optional[str] = None
        self._background: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        try:
            import shap  # type: ignore

            self._shap = shap
            shap_logger = logging.getLogger("shap")
            lib_level = self._log_cfg.get("library_level") or self._expl_cfg.get(
                "shap_library_log_level"
            )
            if lib_level:
                numeric_level = getattr(logging, str(lib_level).upper(), None)
                if isinstance(numeric_level, int):
                    shap_logger.setLevel(numeric_level)
        except Exception:
            self._shap = None
            self.logger.warning("`shap` not available; will use permutation fallback.")
            return

        X_np, _ = self._coerce_X_y(X, None)
        self._is_tree = self._is_tree_model()

        if self._is_tree:
            self._explainer = self._shap.TreeExplainer(self._underlying_model())
            self._explainer_type = "tree"
            return

        explainer_type = str(self._expl_cfg.get("shap_explainer_type", "kernel")).strip().lower()
        if explainer_type in {"kernelexplainer", "kernel"}:
            explainer_type = "kernel"
        elif explainer_type in {"samplingexplainer", "sampling"}:
            explainer_type = "sampling"
        else:
            raise ValueError(
                "experiment.explanation.shap_explainer_type must be one of: "
                "'kernel'|'KernelExplainer'|'sampling'|'SamplingExplainer'. "
                f"Got {self._expl_cfg.get('shap_explainer_type')!r}."
            )

        require_client_local_background(self._expl_cfg)
        self._background = sample_client_local_background(
            X_np,
            expl_cfg=self._expl_cfg,
            random_state=self.random_state,
        )
        predict_fn = self._kernel_predict_fn()
        if explainer_type == "sampling":
            self._explainer = self._shap.SamplingExplainer(predict_fn, self._background)
            self._explainer_type = "sampling"
        else:
            self._explainer = self._shap.KernelExplainer(predict_fn, self._background)
            self._explainer_type = "kernel"

    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        if self._shap is None or self._explainer is None:
            return self._explain_with_permutation(instance)

        inst2d = self._to_numpy_2d(instance)
        pred, t_pred = self._timeit(self._predict, inst2d)
        proba = self._predict_proba(inst2d)
        target_indices = self._resolve_target_indices(pred, proba)
        target_index = None if target_indices is None else int(target_indices[0])

        shap_vals_raw, t_shap = self._timeit(self._shap_values, inst2d)
        shap_vals = self._select_shap_values(shap_vals_raw, target_indices=target_indices)
        expected = self._explainer.expected_value
        exp_val = self._select_expected_value(expected, target_index=target_index)
        proba_value = proba[0] if proba is not None and len(proba) else None
        metadata = {
            "expected_value": exp_val,
            **(
                {
                    "background_data_source": self._expl_cfg.get(
                        "background_data_source",
                        "client_local_train",
                    ),
                    "background_sample_size": int(self._background.shape[0]),
                }
                if self._background is not None
                else {}
            ),
            **(
                {"explained_class": int(target_index)}
                if target_index is not None
                else {}
            ),
        }

        feature_names = self._infer_feature_names(inst2d[0])
        result = self._standardize_explanation_output(
            attributions=np.asarray(shap_vals[0]).tolist()
            if shap_vals.ndim == 2
            else np.asarray(shap_vals).tolist(),
            instance=inst2d[0],
            prediction=self._prediction_output_value(pred),
            prediction_proba=proba_value,
            feature_names=feature_names,
            metadata=metadata,
            per_instance_time=t_pred + t_shap,
        )
        return result

    def explain_batch(self, X: ArrayLike) -> List[Dict[str, Any]]:
        X_np, _ = self._coerce_X_y(X, None)

        if len(X_np) == 0:
            return []

        if self._shap is None or self._explainer is None:
            return super().explain_batch(X_np)

        batch_start = time.time()
        preds = np.asarray(self._predict(X_np), dtype=object)
        proba = self._predict_proba(X_np)
        target_indices = self._resolve_target_indices(preds, proba)
        shap_vals_raw = self._shap_values(X_np)
        shap_vals = self._select_shap_values(shap_vals_raw, target_indices=target_indices)
        expected = self._explainer.expected_value

        results: List[Dict[str, Any]] = []
        for idx in range(len(X_np)):
            instance = X_np[idx]
            pred_val = self._prediction_output_value(preds[idx])
            proba_val = None
            if proba is not None:
                proba_val = proba[idx] if proba.ndim > 1 else proba
            target_index = None if target_indices is None else int(target_indices[idx])
            exp_val = self._select_expected_value(expected, target_index=target_index)
            attr_row = shap_vals[idx] if shap_vals.ndim > 1 else shap_vals
            metadata = {
                "expected_value": exp_val,
                **(
                    {
                        "background_data_source": self._expl_cfg.get(
                            "background_data_source",
                            "client_local_train",
                        ),
                        "background_sample_size": int(self._background.shape[0]),
                    }
                    if self._background is not None
                    else {}
                ),
                **(
                    {"explained_class": int(target_index)}
                    if target_index is not None
                    else {}
                ),
            }
            results.append(
                self._standardize_explanation_output(
                    attributions=np.asarray(attr_row).tolist(),
                    instance=instance,
                    prediction=pred_val,
                    prediction_proba=proba_val,
                    feature_names=self._infer_feature_names(instance),
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
                "shap_explainer_type": self._explainer_type,
                "background_data_source": self._expl_cfg.get(
                    "background_data_source",
                    "client_local_train",
                ),
                "background_sample_size": (
                    None if self._background is None else int(self._background.shape[0])
                ),
            }
        )
        return info

    def _shap_values(self, X: np.ndarray):
        if self._explainer is None:
            raise RuntimeError("SHAP explainer not initialized; call fit() first.")

        kwargs: Dict[str, Any] = {}
        n_features = int(np.asarray(X).shape[1]) if np.asarray(X).ndim >= 2 else 0
        nsamples_int: Optional[int] = None
        nsamples = self._expl_cfg.get("shap_nsamples")
        if nsamples is not None:
            if self._explainer_type in {"kernel", "sampling"}:
                nsamples_int = int(nsamples)
                kwargs["nsamples"] = nsamples_int
            else:
                self.logger.debug(
                    "Ignoring experiment.explanation.shap_nsamples=%r because explainer_type=%r.",
                    nsamples,
                    self._explainer_type,
                )

        def _safe_num_features_reg(*, limit_by_nsamples: bool) -> str:
            k_value = max(1, min(10, max(1, n_features)))
            if limit_by_nsamples and nsamples_int is not None:
                k_value = min(k_value, max(1, nsamples_int - 1))
            return f"num_features({k_value})"

        l1_reg = self._expl_cfg.get("shap_l1_reg")
        if l1_reg is not None:
            if self._explainer_type == "kernel":
                if isinstance(l1_reg, str):
                    token = l1_reg.strip()
                    lowered = token.lower()
                    if lowered in {"auto", "aic", "bic"}:
                        if nsamples_int is not None and nsamples_int <= max(1, n_features):
                            safe = _safe_num_features_reg(limit_by_nsamples=True)
                            self.logger.info(
                                "Overriding shap_l1_reg=%r -> %r because nsamples=%s <= n_features=%s.",
                                lowered,
                                safe,
                                nsamples_int,
                                n_features,
                            )
                            kwargs["l1_reg"] = safe
                        else:
                            kwargs["l1_reg"] = lowered
                    elif lowered == "num_features":
                        k_raw = self._expl_cfg.get("shap_l1_reg_k")
                        if k_raw is None:
                            rng = np.random.default_rng(self.random_state)
                            k_value = int(rng.integers(low=1, high=max(2, n_features + 1)))
                        else:
                            k_value = int(k_raw)
                        k_value = max(1, min(k_value, max(1, n_features)))
                        if nsamples_int is not None:
                            k_value = min(k_value, max(1, nsamples_int - 1))
                        kwargs["l1_reg"] = f"num_features({k_value})"
                    else:
                        kwargs["l1_reg"] = token
                else:
                    kwargs["l1_reg"] = l1_reg
            else:
                self.logger.debug(
                    "Ignoring experiment.explanation.shap_l1_reg=%r because explainer_type=%r.",
                    l1_reg,
                    self._explainer_type,
                )

        def _call_shap_values(call_kwargs: Dict[str, Any]):
            try:
                return self._explainer.shap_values(X, silent=True, **call_kwargs)
            except TypeError:
                return self._explainer.shap_values(X, **call_kwargs)

        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                out = _call_shap_values(kwargs)

            if self._explainer_type == "kernel" and captured:
                try:
                    from sklearn.exceptions import ConvergenceWarning  # type: ignore
                except Exception:  # pragma: no cover
                    ConvergenceWarning = Warning  # type: ignore[misc,assignment]

                saw_singular = False
                for warning in captured:
                    msg = str(getattr(warning, "message", ""))
                    cat = getattr(warning, "category", None)
                    if "Linear regression equation is singular" in msg:
                        saw_singular = True
                        break
                    if cat is not None and issubclass(cat, ConvergenceWarning):
                        if "Regressors in active set degenerate" in msg:
                            saw_singular = True
                            break

                if saw_singular:
                    safe = _safe_num_features_reg(limit_by_nsamples=True)
                    if kwargs.get("l1_reg") != safe:
                        retry_kwargs = dict(kwargs)
                        retry_kwargs["l1_reg"] = safe
                        self.logger.warning(
                            "KernelSHAP emitted singular/degenerate regression warnings; retrying with l1_reg=%r.",
                            safe,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            return _call_shap_values(retry_kwargs)
            return out
        except ValueError as exc:
            msg = str(exc)
            if (
                self._explainer_type == "kernel"
                and "LassoLarsIC" in msg
                and (
                    "number of samples is smaller than the number of features" in msg
                    or "samples is smaller than the number of features" in msg
                )
            ):
                fallback_kwargs = dict(kwargs)
                fallback_kwargs["l1_reg"] = _safe_num_features_reg(limit_by_nsamples=True)
                self.logger.warning(
                    "KernelSHAP failed with LassoLarsIC; retrying with l1_reg=%r.",
                    fallback_kwargs["l1_reg"],
                )
                return _call_shap_values(fallback_kwargs)
            raise

    def _underlying_model(self):
        return getattr(self.model, "model", self.model)

    def _is_tree_model(self) -> bool:
        name = type(self._underlying_model()).__name__.lower()
        return any(
            key in name
            for key in ["decisiontree", "randomforest", "gradientboost", "xgb", "lgbm", "lightgbm"]
        )

    def _kernel_predict_fn(self):
        def _coerce_2d(arr: Any) -> np.ndarray:
            out = np.asarray(arr)
            if out.ndim == 1:
                out = out.reshape(1, -1)
            return out

        if hasattr(self.model, "predict_proba"):
            labels = self._class_labels()
            class_count = len(labels) if labels is not None else None
            target_idx = None
            explicit_target = self._expl_cfg.get("shap_target_class")
            if explicit_target is not None:
                target_idx = self._target_index_from_value(
                    explicit_target,
                    class_count=class_count,
                )
            elif class_count == 2:
                target_idx = 1
            if target_idx is not None:

                def _predict_proba_class(x, idx=target_idx):
                    x2d = _coerce_2d(x)
                    if x2d.shape[0] == 0:
                        return np.empty((0,), dtype=float)
                    return self.model.predict_proba(x2d)[:, idx]

                return _predict_proba_class

            def _predict_proba_all(x):
                x2d = _coerce_2d(x)
                if x2d.shape[0] == 0:
                    n_classes = len(getattr(self.model, "classes_", []) or [])
                    return np.empty((0, n_classes), dtype=float)
                return self.model.predict_proba(x2d)

            return _predict_proba_all

        def _predict_numeric(x):
            x2d = _coerce_2d(x)
            if x2d.shape[0] == 0:
                return np.empty((0,), dtype=float)
            return self._predict_numeric(x2d)

        return _predict_numeric

    def _select_shap_values(
        self,
        shap_values_raw,
        *,
        target_indices: Optional[np.ndarray],
    ) -> np.ndarray:
        if isinstance(shap_values_raw, list):
            if len(shap_values_raw) == 1:
                return np.asarray(shap_values_raw[0])
            if target_indices is None:
                if len(shap_values_raw) == 2:
                    return np.asarray(shap_values_raw[1])
                raise ValueError("SHAP target class could not be resolved for multi-output values.")
            indices = np.asarray(target_indices, dtype=int).ravel()
            return np.vstack([np.asarray(shap_values_raw[c])[i] for i, c in enumerate(indices)])

        shap_values = np.asarray(shap_values_raw)

        if shap_values.ndim == 3:
            if target_indices is None:
                if shap_values.shape[2] == 2:
                    return shap_values[:, :, 1]
                raise ValueError("SHAP target class could not be resolved for multi-output values.")
            indices = np.asarray(target_indices, dtype=int).ravel()
            return np.vstack([shap_values[i, :, indices[i]] for i in range(len(indices))])

        return shap_values

    def _select_expected_value(self, expected_value, *, target_index: Optional[int]) -> float:
        if isinstance(expected_value, (list, np.ndarray)):
            ev = np.asarray(expected_value)
            if ev.ndim == 0:
                return float(ev)
            if target_index is not None and 0 <= int(target_index) < ev.size:
                return float(ev[int(target_index)])
            if ev.size == 2:
                return float(ev[1])
            raise ValueError("SHAP expected_value target class could not be resolved.")
        return float(expected_value)

    def _resolve_target_indices(
        self,
        predictions: np.ndarray,
        prediction_proba: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        explicit_target = self._expl_cfg.get("shap_target_class")
        labels = self._class_labels()
        class_count = len(labels) if labels is not None else None

        if prediction_proba is not None:
            proba = np.asarray(prediction_proba, dtype=float)
            if proba.ndim == 0:
                proba = proba.reshape(1, 1)
            elif proba.ndim == 1:
                proba = proba.reshape(1, -1)
            if class_count is None and proba.ndim > 1:
                class_count = int(proba.shape[1])
        else:
            proba = None

        preds = np.asarray(predictions, dtype=object)
        if preds.ndim == 0:
            preds = preds.reshape(1)
        elif preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.ravel()
        n_rows = int(proba.shape[0]) if proba is not None else int(preds.shape[0])

        if explicit_target is not None:
            target_index = self._target_index_from_value(explicit_target, class_count=class_count)
            return np.full(n_rows, target_index, dtype=int)

        if class_count == 2:
            return np.ones(n_rows, dtype=int)

        if proba is not None and proba.shape[1] > 0:
            return np.argmax(proba, axis=1).astype(int)

        if class_count is not None:
            return self._prediction_indices(preds)
        return None

    def _target_index_from_value(self, target_value: Any, *, class_count: Optional[int]) -> int:
        labels = self._class_labels()
        if labels is not None:
            for idx, label in enumerate(labels):
                if label == target_value:
                    return idx

        try:
            target_index = int(target_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"experiment.explanation.shap_target_class={target_value!r} could not be resolved."
            ) from exc

        if class_count is not None and not 0 <= target_index < class_count:
            raise ValueError(
                f"experiment.explanation.shap_target_class={target_value!r} is out of bounds "
                f"for {class_count} classes."
            )
        return target_index

    def _explain_with_permutation(self, instance: InstanceLike) -> Dict[str, Any]:
        inst = self._to_numpy_2d(instance)[0]
        bg_mean = getattr(self.dataset, "feature_means", None)
        if bg_mean is None:
            X_bg = getattr(self.dataset, "X_train", None)
            if X_bg is not None:
                bg_mean = np.mean(np.asarray(X_bg), axis=0)
            else:
                bg_mean = np.zeros_like(inst)

        base_pred = float(np.asarray(self._predict_numeric(inst)).ravel()[0])
        importances = np.zeros_like(inst, dtype=float)

        for j in range(len(inst)):
            perturbed = inst.copy()
            perturbed[j] = bg_mean[j]
            new_pred = float(np.asarray(self._predict_numeric(perturbed)).ravel()[0])
            importances[j] = base_pred - new_pred

        feature_names = self._infer_feature_names(inst)
        return self._standardize_explanation_output(
            attributions=importances.tolist(),
            instance=inst,
            prediction=base_pred,
            prediction_proba=None,
            feature_names=feature_names,
            metadata={"fallback": "permutation"},
            per_instance_time=0.0,
        )
