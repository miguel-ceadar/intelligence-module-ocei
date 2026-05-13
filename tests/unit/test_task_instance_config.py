"""Tests for the per-kind task instance config schemas.

These cover parsing only — the runtime path that consumes them lands in
``build_registry_from_config`` after the kind builders ship.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from intelligence.config.settings import (
    ArimaTaskConfig,
    DriftTaskConfig,
    LstmTaskConfig,
    TaskInstanceConfig,
    XgbTaskConfig,
)

_adapter = TypeAdapter(TaskInstanceConfig)


def _parse(payload: dict):
    return _adapter.validate_python(payload)


def test_arima_minimal_block_parses():
    cfg = _parse({"kind": "arima", "feature": "cpu"})
    assert isinstance(cfg, ArimaTaskConfig)
    assert cfg.feature == "cpu"
    assert cfg.steps_back == 1
    assert cfg.model_params.p == 5
    assert cfg.query is None


def test_arima_with_full_overrides_parses():
    cfg = _parse(
        {
            "kind": "arima",
            "feature": "mem",
            "value_range": [0.0, 1.0],
            "steps_back": 1,
            "query": "avg(rate(foo[1m]))",
            "model_params": {"p": 2, "d": 1, "q": 1},
        }
    )
    assert isinstance(cfg, ArimaTaskConfig)
    assert cfg.value_range == (0.0, 1.0)
    assert cfg.model_params.p == 2
    assert cfg.model_params.q == 1


def test_xgb_block_parses_with_defaults():
    cfg = _parse({"kind": "xgb", "feature": "cpu"})
    assert isinstance(cfg, XgbTaskConfig)
    assert cfg.steps_back == 6
    assert cfg.model_params.n_estimators == 100
    assert cfg.model_params.eta == pytest.approx(0.1)


def test_xgb_model_params_accept_unknown_xgboost_field():
    # XgbModelParams has extra="allow" so forward-compat fields don't
    # require library updates whenever xgboost adds a knob.
    cfg = _parse(
        {
            "kind": "xgb",
            "feature": "cpu",
            "model_params": {"n_estimators": 50, "subsample": 0.8},
        }
    )
    assert isinstance(cfg, XgbTaskConfig)
    assert cfg.model_params.model_extra == {"subsample": 0.8}


def test_lstm_block_carries_batch_size_and_network_shape():
    cfg = _parse(
        {
            "kind": "lstm",
            "feature": "cpu",
            "batch_size": 32,
            "model_params": {"hidden_size": 16, "num_epochs": 10},
        }
    )
    assert isinstance(cfg, LstmTaskConfig)
    assert cfg.batch_size == 32
    assert cfg.model_params.hidden_size == 16
    assert cfg.model_params.num_epochs == 10
    assert cfg.model_params.input_size == 1  # default preserved


def test_lstm_horizon_defaults_to_one():
    """Wave 1 #2: LSTM tasks carry a top-level ``horizon`` field that
    becomes both ``output_size`` (at train) and ``max_horizon`` (in the
    contract). Defaults to 1 so single-step deployments stay unchanged."""
    if "horizon" not in LstmTaskConfig.model_fields:
        pytest.skip("LstmTaskConfig.horizon not implemented yet")
    cfg = _parse({"kind": "lstm", "feature": "cpu"})
    assert isinstance(cfg, LstmTaskConfig)
    assert cfg.horizon == 1


def test_lstm_horizon_overrides_default():
    if "horizon" not in LstmTaskConfig.model_fields:
        pytest.skip("LstmTaskConfig.horizon not implemented yet")
    cfg = _parse({"kind": "lstm", "feature": "cpu", "horizon": 6})
    assert isinstance(cfg, LstmTaskConfig)
    assert cfg.horizon == 6


def test_drift_requires_forecaster_reference():
    cfg = _parse(
        {
            "kind": "drift",
            "feature": "cpu",
            "forecaster": "cpu_forecast_arima",
        }
    )
    assert isinstance(cfg, DriftTaskConfig)
    assert cfg.forecaster == "cpu_forecast_arima"
    assert cfg.chunk_size == 12
    assert cfg.metric == "jensen_shannon"


def test_drift_without_forecaster_fails():
    with pytest.raises(ValidationError) as exc:
        _parse({"kind": "drift", "feature": "cpu"})
    assert "forecaster" in str(exc.value)


def test_unknown_kind_fails_loudly():
    with pytest.raises(ValidationError) as exc:
        _parse({"kind": "transformer", "feature": "cpu"})
    # Pydantic reports the kind as the discriminator that didn't match.
    assert "transformer" in str(exc.value)


def test_missing_feature_fails():
    with pytest.raises(ValidationError) as exc:
        _parse({"kind": "arima"})
    assert "feature" in str(exc.value)


def test_value_range_accepts_open_intervals():
    # Optional — leaving value_range off is fine for non-fractional metrics.
    cfg = _parse({"kind": "arima", "feature": "request_rate"})
    assert cfg.value_range is None


def test_bootstrap_block_nested_under_task():
    cfg = _parse(
        {
            "kind": "arima",
            "feature": "cpu",
            "bootstrap": {
                "auto_train_on_startup": True,
                "window": "24h",
                "step": "1m",
            },
        }
    )
    assert isinstance(cfg, ArimaTaskConfig)
    assert cfg.bootstrap.auto_train_on_startup is True
    assert cfg.bootstrap.window == "24h"
    assert cfg.bootstrap.step == "1m"
