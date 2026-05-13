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
    FeatureSpec,
    LstmTaskConfig,
    TaskInstanceConfig,
    XgbTaskConfig,
)

_adapter = TypeAdapter(TaskInstanceConfig)


def _parse(payload: dict):
    return _adapter.validate_python(payload)


def test_arima_minimal_block_parses():
    cfg = _parse({"kind": "arima", "features": [{"name": "cpu"}]})
    assert isinstance(cfg, ArimaTaskConfig)
    assert cfg.features == [FeatureSpec(name="cpu")]
    assert cfg.steps_back == 1
    assert cfg.model_params.p == 5
    assert cfg.features[0].query is None


def test_arima_with_full_overrides_parses():
    cfg = _parse(
        {
            "kind": "arima",
            "features": [
                {
                    "name": "mem",
                    "value_range": [0.0, 1.0],
                    "query": "avg(rate(foo[1m]))",
                }
            ],
            "steps_back": 1,
            "model_params": {"p": 2, "d": 1, "q": 1},
        }
    )
    assert isinstance(cfg, ArimaTaskConfig)
    assert cfg.features[0].value_range == (0.0, 1.0)
    assert cfg.features[0].query == "avg(rate(foo[1m]))"
    assert cfg.model_params.p == 2
    assert cfg.model_params.q == 1


def test_xgb_block_parses_with_defaults():
    cfg = _parse({"kind": "xgb", "features": [{"name": "cpu"}]})
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
            "features": [{"name": "cpu"}],
            "model_params": {"n_estimators": 50, "subsample": 0.8},
        }
    )
    assert isinstance(cfg, XgbTaskConfig)
    assert cfg.model_params.model_extra == {"subsample": 0.8}


def test_lstm_block_carries_batch_size_and_network_shape():
    cfg = _parse(
        {
            "kind": "lstm",
            "features": [{"name": "cpu"}],
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
    cfg = _parse({"kind": "lstm", "features": [{"name": "cpu"}]})
    assert isinstance(cfg, LstmTaskConfig)
    assert cfg.horizon == 1


def test_lstm_horizon_overrides_default():
    if "horizon" not in LstmTaskConfig.model_fields:
        pytest.skip("LstmTaskConfig.horizon not implemented yet")
    cfg = _parse({"kind": "lstm", "features": [{"name": "cpu"}], "horizon": 6})
    assert isinstance(cfg, LstmTaskConfig)
    assert cfg.horizon == 6


def test_drift_requires_forecaster_reference():
    cfg = _parse(
        {
            "kind": "drift",
            "features": [{"name": "cpu"}],
            "forecaster": "cpu_forecast_arima",
        }
    )
    assert isinstance(cfg, DriftTaskConfig)
    assert cfg.forecaster == "cpu_forecast_arima"
    assert cfg.chunk_size == 12
    assert cfg.metric == "jensen_shannon"


def test_drift_without_forecaster_fails():
    with pytest.raises(ValidationError) as exc:
        _parse({"kind": "drift", "features": [{"name": "cpu"}]})
    assert "forecaster" in str(exc.value)


def test_unknown_kind_fails_loudly():
    with pytest.raises(ValidationError) as exc:
        _parse({"kind": "transformer", "features": [{"name": "cpu"}]})
    # Pydantic reports the kind as the discriminator that didn't match.
    assert "transformer" in str(exc.value)


def test_missing_features_fails():
    with pytest.raises(ValidationError) as exc:
        _parse({"kind": "arima"})
    assert "features" in str(exc.value)


def test_empty_features_list_fails():
    """``features`` must be non-empty — a task with no features makes no sense."""
    with pytest.raises(ValidationError) as exc:
        _parse({"kind": "arima", "features": []})
    assert "features" in str(exc.value).lower()


def test_value_range_accepts_open_intervals():
    # Optional — leaving value_range off is fine for non-fractional metrics.
    cfg = _parse({"kind": "arima", "features": [{"name": "request_rate"}]})
    assert cfg.features[0].value_range is None


def test_multivariate_features_list_parses():
    """The schema accepts more than one feature; the first is the
    target, the rest are covariates. Per-kind multivariate behavior
    lands in later gates — this just verifies the schema shape."""
    cfg = _parse(
        {
            "kind": "xgb",
            "features": [
                {"name": "cpu", "query": "avg(rate(node_cpu[30s]))", "value_range": [0.0, 1.0]},
                {"name": "memory", "query": "avg(node_mem)", "value_range": [0.0, 1.0]},
                {"name": "load", "query": "avg(node_load1)"},
            ],
        }
    )
    assert isinstance(cfg, XgbTaskConfig)
    assert len(cfg.features) == 3
    assert [f.name for f in cfg.features] == ["cpu", "memory", "load"]
    assert cfg.features[2].value_range is None


def test_bootstrap_block_nested_under_task():
    cfg = _parse(
        {
            "kind": "arima",
            "features": [{"name": "cpu"}],
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
