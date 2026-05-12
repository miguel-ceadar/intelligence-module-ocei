"""Train + predict roundtrip for ``XgbModel``.

Uses a synthetic random walk so the test is sub-second and
deterministic. Verifies:
  - ``make_xgb_prepare`` produces the components shape ``train_xgb`` expects.
  - ``XgbModel.train`` saves a Bento with the X and y scalers populated.
  - ``XgbModel.predict`` returns a float from the saved Bento using a
    window of observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_cpu(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(n) * 0.02) + 0.5
    return pd.DataFrame({"timestamp": np.arange(n), "cpu": walk.clip(0.05, 0.95)})


def test_make_xgb_prepare_yields_expected_components():
    from intelligence.ml.models.xgb import make_xgb_prepare

    prep = make_xgb_prepare(look_back=4, num_variables=1)
    comps = prep(_synthetic_cpu())

    for key in ("X_train", "X_test", "y_train", "y_test", "scaler_obj", "scaler_X", "look_back"):
        assert key in comps, f"missing component: {key}"
    assert comps["look_back"] == 4
    assert comps["X_train"].shape[1] == 4  # look_back lag features
    assert hasattr(comps["scaler_X"], "transform")
    assert hasattr(comps["scaler_obj"], "transform")


def test_xgb_train_then_predict_roundtrip(tmp_path, monkeypatch):
    from intelligence.ml.models.xgb import XgbModel, make_xgb_prepare

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    prep = make_xgb_prepare(look_back=6, num_variables=1)
    comps = prep(_synthetic_cpu(n=200))
    comps["model_parameters"] = {"n_estimators": 20, "max_depth": 3, "eta": 0.1}

    model = XgbModel()
    bento, metrics = model.train(comps, bento_name="t_xgb_roundtrip", extras=None)

    assert "mae" in metrics
    # Bento has both scalers + look_back saved.
    assert "scaler_X" in bento.custom_objects
    assert "scaler_obj" in bento.custom_objects
    assert bento.custom_objects["look_back"] == 6

    # Predict from a window — should return a float.
    window_values = _synthetic_cpu(n=8).iloc[-6:]["cpu"].tolist()
    yhat = model.predict(bento, {"cpu": window_values})
    assert isinstance(yhat, float)
    # The synthetic data is clipped to [0.05, 0.95]; predictions in that
    # neighbourhood are sane. Loose bound — we're not asserting accuracy.
    assert -1.0 < yhat < 2.0


def test_xgb_predict_rejects_short_window(tmp_path, monkeypatch):
    from intelligence.ml.models.xgb import XgbModel, make_xgb_prepare

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    prep = make_xgb_prepare(look_back=6, num_variables=1)
    comps = prep(_synthetic_cpu(n=200))
    comps["model_parameters"] = {"n_estimators": 10}
    model = XgbModel()
    bento, _metrics = model.train(comps, bento_name="t_xgb_short", extras=None)

    with pytest.raises(ValueError, match="at least 6"):
        model.predict(bento, {"cpu": [0.5, 0.5]})
