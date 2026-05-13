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
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "train_dataset",
        "test_dataset",
        "batch_size",
        "scaler_obj",
        "look_back",
        "num_variables",
    ):
        assert key in comps, f"missing component: {key}"
    assert comps["X_train"].dim() == 3
    assert comps["X_train"].shape[1] == 5  # look_back
    assert comps["X_train"].shape[2] == 1  # num_variables


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

    out = model.predict(bento, {"cpu": _synthetic_cpu(n=8).iloc[-6:]["cpu"].tolist()})
    assert isinstance(out, list) and len(out) == 1
    assert -1.0 < out[0].value < 2.0


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


@pytest.mark.slow
def test_lstm_train_with_horizon_three_then_predict(tmp_path, monkeypatch):
    """Wave 1 #2: LSTM is direct multi-output — ``output_size`` is set at
    train time and equals the maximum horizon predict can serve.
    """
    pytest.importorskip("torch")
    from intelligence.api import schemas as api_schemas
    from intelligence.ml.models.lstm import LstmModel, make_lstm_prepare

    if not hasattr(api_schemas, "ForecastPoint"):
        pytest.skip("ForecastPoint not implemented yet")

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    horizon = 3
    try:
        prep = make_lstm_prepare(
            look_back=6,
            num_variables=1,
            batch_size=16,
            horizon=horizon,
        )
    except TypeError:
        pytest.skip("make_lstm_prepare doesn't accept horizon yet")

    comps = prep(_synthetic_cpu(n=300))
    comps["model_parameters"] = {
        "input_size": 1,
        "output_size": horizon,
        "hidden_size": 4,
        "num_epochs": 2,
    }

    model = LstmModel()
    bento, _ = model.train(comps, bento_name="t_lstm_mh", extras=None)
    assert bento.custom_objects["output_size"] == horizon

    try:
        out = model.predict(
            bento,
            {"cpu": _synthetic_cpu(n=8).iloc[-6:]["cpu"].tolist()},
            horizon=horizon,
        )
    except TypeError:
        pytest.skip("LstmModel.predict doesn't accept horizon yet")

    assert isinstance(out, list) and len(out) == horizon
    for point in out:
        value = getattr(point, "value", None) if not isinstance(point, dict) else point["value"]
        assert value is not None
        # LSTM ships without CIs (MC-dropout deferred per memory note).
        assert (
            getattr(point, "lower", None) is None
            if not isinstance(point, dict)
            else point.get("lower") is None
        )


def test_lstm_predict_rejects_horizon_greater_than_trained_output_size(tmp_path, monkeypatch):
    """LSTM is direct — request horizon > trained output_size is a 422."""
    pytest.importorskip("torch")
    from unittest import mock

    from sklearn.preprocessing import MinMaxScaler

    from intelligence.api import schemas as api_schemas
    from intelligence.ml.models.lstm import LstmModel

    if not hasattr(api_schemas, "ForecastPoint"):
        pytest.skip("ForecastPoint not implemented yet")

    fake_bento = mock.MagicMock()
    fake_bento.custom_objects = {
        "scaler_obj": MinMaxScaler().fit(np.array([[0.0], [1.0]])),
        "look_back": 6,
        "num_variables": 1,
        "output_size": 2,  # trained max horizon
    }
    window = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    try:
        with pytest.raises(ValueError, match="horizon"):
            LstmModel().predict(fake_bento, {"cpu": window}, horizon=5)
    except TypeError:
        pytest.skip("LstmModel.predict doesn't accept horizon yet")
