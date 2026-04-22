from __future__ import annotations

from typing import Any, Dict

from .causal_shap_explainer import CausalSHAPExplainer
from .integrated_gradients_explainer import IntegratedGradientsExplainer
from .lime_explainer import LIMEExplainer
from .shap_explainer import SHAPExplainer

_NAME2CLS = {
    "causal_shap": CausalSHAPExplainer,
    "integrated_gradients": IntegratedGradientsExplainer,
    "lime": LIMEExplainer,
    "shap": SHAPExplainer,
}


def make_explainer(config: Dict[str, Any], model: Any, dataset: Any):
    typ = (config.get("type") or "shap").lower()
    if typ not in _NAME2CLS:
        raise ValueError(f"Unknown explainer type: {typ}")
    return _NAME2CLS[typ](config=config, model=model, dataset=dataset)
