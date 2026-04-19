from __future__ import annotations

import json

import numpy as np

from fed_perso_xai.data.partitioning import ClientSplit
from fed_perso_xai.data.serialization import load_client_datasets, save_federated_dataset


def test_dataset_serialization_layout_and_loading(tmp_path) -> None:
    client_splits = [
        ClientSplit(
            client_id=0,
            X_train=np.ones((4, 3)),
            y_train=np.asarray([0, 1, 0, 1]),
            X_test=np.ones((2, 3)) * 2,
            y_test=np.asarray([0, 1]),
        ),
        ClientSplit(
            client_id=1,
            X_train=np.ones((5, 3)) * 3,
            y_train=np.asarray([1, 1, 0, 0, 1]),
            X_test=np.ones((2, 3)) * 4,
            y_test=np.asarray([1, 0]),
        ),
    ]
    artifacts = save_federated_dataset(
        dataset_name="adult_income",
        output_root=tmp_path / "datasets",
        num_clients=2,
        alpha=1.0,
        seed=42,
        feature_names=["f0", "f1", "f2"],
        preprocessing_info={"feature_names": ["f0", "f1", "f2"]},
        client_splits=client_splits,
        global_eval=(np.zeros((3, 3)), np.asarray([0, 1, 0])),
    )

    assert (artifacts.root_dir / "client_0" / "train.npz").exists()
    assert (artifacts.root_dir / "client_0" / "test.npz").exists()
    assert (artifacts.root_dir / "client_1" / "train.npz").exists()
    assert (artifacts.root_dir / "client_1" / "test.npz").exists()
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset_name"] == "adult_income"
    assert metadata["feature_count"] == 3

    loaded = load_client_datasets(artifacts.root_dir, num_clients=2)
    assert loaded[0].X_train.shape == (4, 3)
    assert loaded[1].X_test.shape == (2, 3)
