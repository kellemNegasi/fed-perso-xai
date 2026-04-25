from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fed_perso_xai.recommender.user_simulation import (
    DirichletPersonaSimulator,
    PersonaConfig,
)


def _persona() -> PersonaConfig:
    return PersonaConfig.from_dict(
        {
            "persona": "unit",
            "type": "flat_dirichlet",
            "tau": 0.05,
            "properties": {
                "quality": {
                    "preference": 1,
                    "metrics": ["quality", "missing_metric"],
                }
            },
        }
    )


def test_dirichlet_persona_labels_metric_z_columns_and_reports_missing_metrics() -> None:
    simulator = DirichletPersonaSimulator(
        _persona(),
        seed=0,
        label_seed=0,
        concentration_c=1.0,
        tau=0.01,
    )
    simulator.metric_weights = {"quality": 1.0, "missing_metric": 0.0}
    candidates = pd.DataFrame(
        {
            "client_id": ["client_000", "client_000"],
            "dataset_index": [7, 7],
            "instance_id": ["row-7", "row-7"],
            "method_variant": ["better", "worse"],
            "metric_quality_z": [5.0, -5.0],
        }
    )

    labels, metadata = simulator.label_client_candidates(candidates)

    assert len(labels) == 1
    assert labels.iloc[0]["pair_1"] == "better"
    assert labels.iloc[0]["pair_2"] == "worse"
    assert int(labels.iloc[0]["label"]) == 0
    assert labels.iloc[0]["probability_pair_1_preferred"] == pytest.approx(1.0)
    assert metadata["active_metrics"] == ["quality"]
    assert metadata["missing_configured_metrics"] == ["missing_metric"]


def test_dirichlet_persona_rejects_context_without_active_metrics() -> None:
    simulator = DirichletPersonaSimulator(_persona(), seed=0, concentration_c=1.0)
    candidates = pd.DataFrame(
        {
            "dataset_index": [0, 0],
            "method_variant": ["a", "b"],
            "some_other_feature": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="No persona metrics"):
        simulator.label_client_candidates(candidates)


def test_dirichlet_persona_dedupes_duplicate_variants() -> None:
    simulator = DirichletPersonaSimulator(_persona(), seed=0, label_seed=0, concentration_c=1.0)
    simulator.metric_weights = {"quality": 1.0, "missing_metric": 0.0}
    candidates = pd.DataFrame(
        {
            "dataset_index": [0, 0, 0],
            "method_variant": ["a", "a", "b"],
            "metric_quality_z": [1.0, 1.0, 0.0],
        }
    )

    labels, _ = simulator.label_client_candidates(candidates)

    assert len(labels) == 1
    assert set(np.ravel(labels[["pair_1", "pair_2"]].to_numpy())) == {"a", "b"}
