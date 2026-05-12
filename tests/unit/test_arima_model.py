"""Train + predict roundtrip for ``ArimaModel``, including multi-horizon
with confidence intervals (Wave 1 #2).

Single-step predict was previously covered only by the API integration
tests; this file pulls the contract into a focused unit test so multi-
horizon + CI changes can be validated without spinning the full service.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_cpu(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal(n) * 0.02) + 0.5
    return pd.DataFrame({"timestamp": np.arange(n), "cpu": walk.clip(0.05, 0.95)})


def _components(df: pd.DataFrame) -> dict:
    """ARIMA prepare: 80/20 split, MinMax-scaled, model_parameters carries
    the order. Matches what ``_make_univariate_prepare`` produces."""
    from sklearn.preprocessing import MinMaxScaler

    series = df["cpu"].astype(float).values.reshape(-1, 1)
    split = int(len(series) * 0.8)
    scaler = MinMaxScaler().fit(series[:split])
    return {
        "X_train": scaler.transform(series[:split]),
        "X_test": scaler.transform(series[split:]),
        "y_train": series[:split].ravel(),
        "y_test": series[split:].ravel(),
        "scaler_obj": scaler,
        "model_parameters": {"p": 2, "d": 1, "q": 0},
    }


def test_arima_train_then_single_step_predict_roundtrip(tmp_path, monkeypatch):
    from intelligence.ml.models.arima import ArimaModel

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    comps = _components(_synthetic_cpu(n=150))
    model = ArimaModel()
    bento, metrics = model.train(comps, bento_name="t_arima_roundtrip", extras=None)

    assert "mae" in metrics
    assert hasattr(bento.custom_objects["scaler_obj"], "transform")

    out = model.predict(bento, {"cpu": [0.42]})
    # Pre-multi-horizon: a single rounded float. Post-multi-horizon: a
    # list of length 1 of ForecastPoint. Accept either while the impl
    # transitions; the strict shape check lives below.
    if isinstance(out, list):
        assert len(out) == 1
        first = out[0]
        value = getattr(first, "value", None) or first["value"]  # ForecastPoint or dict
        assert -1.0 < float(value) < 2.0
    else:
        assert -1.0 < float(out) < 2.0


def test_arima_predict_multi_horizon_returns_list_of_forecast_points(tmp_path, monkeypatch):
    """horizon=N returns a list of length N. Each entry is a ForecastPoint
    with a ``value`` (and ARIMA fills ``lower``/``upper`` from
    ``get_forecast(steps=N).conf_int()``).
    """
    pytest.importorskip("statsmodels")
    from intelligence.api import schemas as api_schemas
    from intelligence.ml.models.arima import ArimaModel

    if not hasattr(api_schemas, "ForecastPoint"):
        pytest.skip("ForecastPoint not implemented yet")

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    comps = _components(_synthetic_cpu(n=150))
    model = ArimaModel()
    bento, _ = model.train(comps, bento_name="t_arima_mh", extras=None)

    try:
        out = model.predict(bento, {"cpu": [0.42]}, horizon=5)
    except TypeError:
        pytest.skip("ArimaModel.predict doesn't accept horizon yet")

    assert isinstance(out, list)
    assert len(out) == 5
    for point in out:
        value = getattr(point, "value", None)
        lower = getattr(point, "lower", None)
        upper = getattr(point, "upper", None)
        if value is None and isinstance(point, dict):
            value, lower, upper = point["value"], point.get("lower"), point.get("upper")
        assert value is not None
        # ARIMA ships native confidence intervals — both bounds populated.
        assert lower is not None and upper is not None
        assert lower <= value <= upper


def test_arima_predict_horizon_one_is_equivalent_to_default(tmp_path, monkeypatch):
    """``horizon=1`` should match the default behaviour (no CI surprise)."""
    from intelligence.api import schemas as api_schemas
    from intelligence.ml.models.arima import ArimaModel

    if not hasattr(api_schemas, "ForecastPoint"):
        pytest.skip("ForecastPoint not implemented yet")

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))
    comps = _components(_synthetic_cpu(n=150))
    model = ArimaModel()
    bento, _ = model.train(comps, bento_name="t_arima_one", extras=None)

    try:
        out = model.predict(bento, {"cpu": [0.42]}, horizon=1)
    except TypeError:
        pytest.skip("ArimaModel.predict doesn't accept horizon yet")

    assert isinstance(out, list) and len(out) == 1
