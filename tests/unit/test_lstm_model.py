"""Train + predict roundtrip for ``LstmModel``.

Uses a synthetic random walk + tiny model (hidden_size=4, 2 epochs) so
the test stays under a couple of seconds. Verifies:
  - ``make_lstm_prepare`` produces 3-D tensors + ``TimeSeriesDataset`` instances.
  - ``LstmModel.train`` saves a Bento with the scaler + window metadata.
  - ``LstmModel.predict`` returns a float from the saved Bento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_cpu(n: int = 200, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(n) * 0.02) + 0.5
    return pd.DataFrame({"timestamp": np.arange(n), "cpu": walk.clip(0.05, 0.95)})


def test_make_lstm_prepare_yields_3d_tensors():
    pytest.importorskip("torch")
    from intelligence.ml.models.lstm import make_lstm_prepare

    prep = make_lstm_prepare(look_back=5, num_variables=1, batch_size=8)
    comps = prep(_synthetic_cpu())

    for key in (
        "X_train", "X_test", "y_train", "y_test",
        "train_dataset", "test_dataset", "batch_size",
        "scaler_obj", "look_back", "num_variables",
    ):
        assert key in comps, f"missing component: {key}"
    assert comps["X_train"].dim() == 3
    assert comps["X_train"].shape[1] == 5    # look_back
    assert comps["X_train"].shape[2] == 1    # num_variables


@pytest.mark.slow
def test_lstm_train_then_predict_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from intelligence.ml.models.lstm import LstmModel, make_lstm_prepare

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    prep = make_lstm_prepare(look_back=6, num_variables=1, batch_size=16)
    comps = prep(_synthetic_cpu(n=300))
    # Tiny model + 2 epochs so the test runs in seconds, not minutes.
    comps["model_parameters"] = {
        "input_size": 1,
        "output_size": 1,
        "hidden_size": 4,
        "num_epochs": 2,
    }

    model = LstmModel()
    bento, metrics = model.train(comps, bento_name="t_lstm_roundtrip", extras=None)

    assert isinstance(metrics, dict) and len(metrics) > 0
    assert bento.custom_objects["look_back"] == 6
    assert bento.custom_objects["num_variables"] == 1
    assert hasattr(bento.custom_objects["scaler_obj"], "transform")

    yhat = model.predict(bento, {"cpu": _synthetic_cpu(n=8).iloc[-6:]["cpu"].tolist()})
    assert isinstance(yhat, float)
    assert -1.0 < yhat < 2.0


def test_lstm_predict_rejects_short_window(tmp_path, monkeypatch):
    """Verified without training — build a fake Bento that carries the
    metadata predict needs. Avoids the slow training path for the
    contract-check tests.
    """
    pytest.importorskip("torch")
    from unittest import mock

    from sklearn.preprocessing import MinMaxScaler

    from intelligence.ml.models.lstm import LstmModel

    fake_bento = mock.MagicMock()
    fake_bento.custom_objects = {
        "scaler_obj": MinMaxScaler().fit(np.array([[0.0], [1.0]])),
        "look_back": 6,
        "num_variables": 1,
    }
    with pytest.raises(ValueError, match="at least 6"):
        LstmModel().predict(fake_bento, {"cpu": [0.5, 0.5]})
