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
    # ``input_size`` / ``output_size`` are deliberately *not* on the schema —
    # the builder derives them from ``len(features)`` and ``horizon``.
    assert "input_size" not in LstmTaskConfig.model_fields["model_params"].annotation.model_fields


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


# --- §4 maintainability: schema tightening ---------------------------------
#
# Each block here pins a class of misconfig that used to parse cleanly and
# blow up (or silently mis-train) at runtime.


def test_lstm_model_params_have_no_dead_input_or_output_size():
    """``LstmModelParams.input_size`` / ``output_size`` are silently
    overridden by the builder (set from ``len(features)`` and
    ``horizon``). They're a trap on the schema — a YAML that sets them
    is lying to its reader. Drop them.
    """
    from intelligence.config.settings import LstmModelParams

    assert "input_size" not in LstmModelParams.model_fields
    assert "output_size" not in LstmModelParams.model_fields
    # ``hidden_size`` is a real knob (passed through by the builder) — keep it.
    assert "hidden_size" in LstmModelParams.model_fields


@pytest.mark.parametrize(
    "kind,field,bad_value",
    [
        ("arima", "steps_back", 0),
        ("arima", "steps_back", -1),
        ("xgb", "steps_back", 0),
        ("lstm", "steps_back", 0),
        ("lstm", "horizon", 0),
        ("lstm", "batch_size", 0),
    ],
)
def test_task_window_fields_reject_non_positive(kind, field, bad_value):
    payload = {"kind": kind, "features": [{"name": "cpu"}], field: bad_value}
    with pytest.raises(ValidationError, match=field):
        _parse(payload)


def test_drift_chunk_size_rejects_non_positive():
    payload = {
        "kind": "drift",
        "features": [{"name": "cpu"}],
        "forecaster": "f",
        "chunk_size": 0,
    }
    with pytest.raises(ValidationError, match="chunk_size"):
        _parse(payload)


@pytest.mark.parametrize(
    "params,bad_field",
    [
        ({"n_estimators": 0}, "n_estimators"),
        ({"max_depth": 0}, "max_depth"),
        ({"eta": 0.0}, "eta"),
    ],
)
def test_xgb_model_params_reject_non_positive(params, bad_field):
    payload = {"kind": "xgb", "features": [{"name": "cpu"}], "model_params": params}
    with pytest.raises(ValidationError, match=bad_field):
        _parse(payload)


@pytest.mark.parametrize(
    "params,bad_field",
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"num_epochs": 0}, "num_epochs"),
    ],
)
def test_lstm_model_params_reject_non_positive(params, bad_field):
    payload = {"kind": "lstm", "features": [{"name": "cpu"}], "model_params": params}
    with pytest.raises(ValidationError, match=bad_field):
        _parse(payload)


@pytest.mark.parametrize(
    "params,bad_field",
    [
        ({"p": -1}, "p"),
        ({"d": -1}, "d"),
        ({"q": -1}, "q"),
    ],
)
def test_arima_model_params_reject_negative(params, bad_field):
    """ARIMA p/d/q must be non-negative; ``(0, 0, 0)`` is degenerate but
    statsmodels' job to reject, not the schema's."""
    payload = {"kind": "arima", "features": [{"name": "cpu"}], "model_params": params}
    with pytest.raises(ValidationError, match=bad_field):
        _parse(payload)


def test_drift_metric_restricted_to_nannyml_continuous_methods():
    """NannyML's ``UnivariateDriftCalculator`` registers exactly four
    methods for continuous features: ``jensen_shannon``,
    ``kolmogorov_smirnov``, ``wasserstein``, ``hellinger``. Anything
    else blows up inside the calculator at predict time — push the
    check up to the schema."""
    for metric in ("jensen_shannon", "kolmogorov_smirnov", "wasserstein", "hellinger"):
        cfg = _parse(
            {
                "kind": "drift",
                "features": [{"name": "cpu"}],
                "forecaster": "f",
                "metric": metric,
            }
        )
        assert isinstance(cfg, DriftTaskConfig)
        assert cfg.metric == metric

    with pytest.raises(ValidationError, match="metric"):
        _parse(
            {
                "kind": "drift",
                "features": [{"name": "cpu"}],
                "forecaster": "f",
                "metric": "kullback_leibler",
            }
        )


def test_feature_value_range_rejects_inverted_bounds():
    """A ``(hi, lo)`` value_range parses cleanly today and every predict
    request then fails the runtime range check. Catch it at schema time."""
    with pytest.raises(ValidationError, match="value_range"):
        _parse(
            {
                "kind": "arima",
                "features": [{"name": "cpu", "value_range": [1.0, 0.0]}],
            }
        )


def test_feature_value_range_rejects_equal_bounds():
    """``lo == hi`` makes the only valid value the single endpoint — a
    spec-time bug, not an intentional constraint. Reject."""
    with pytest.raises(ValidationError, match="value_range"):
        _parse(
            {
                "kind": "arima",
                "features": [{"name": "cpu", "value_range": [0.5, 0.5]}],
            }
        )


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
