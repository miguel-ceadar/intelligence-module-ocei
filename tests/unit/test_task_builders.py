"""Tests for per-kind task builders.

Verifies that each builder produces a ``BaseTask`` instance whose
public surface matches what today's hand-rolled factories produce —
same task name, same InputSpec, same model class/params.
"""

from __future__ import annotations

import pytest

from intelligence.config.settings import (
    AppConfig,
    ArimaTaskConfig,
    DriftTaskConfig,
    IntelligenceConfig,
    LstmTaskConfig,
    PrometheusConfig,
    TelemetryConfig,
    XgbTaskConfig,
)
from intelligence.tasks.builders import (
    BUILDERS,
    build_arima_task,
    build_drift_task,
    build_lstm_task,
    build_xgb_task,
    get_builder,
)


@pytest.fixture
def static_cfg() -> IntelligenceConfig:
    return IntelligenceConfig()


@pytest.fixture
def prom_cfg() -> IntelligenceConfig:
    return IntelligenceConfig(
        telemetry=TelemetryConfig(
            source="prometheus",
            prometheus=PrometheusConfig(endpoint="http://prom.example:9090"),
        ),
    )


def test_arima_builder_constructs_basetask_with_arima_model(static_cfg):
    task_cfg = ArimaTaskConfig(
        kind="arima",
        feature="cpu",
        value_range=(0.0, 1.0),
        steps_back=1,
    )
    task = build_arima_task("cpu_forecast_arima", task_cfg, static_cfg)
    assert task.name == "cpu_forecast_arima"
    assert task.model is not None
    assert task.model.name == "arima"
    assert task.model.default_params == {"p": 5, "d": 1, "q": 0}
    assert task.input_spec.n_features == 1
    assert task.input_spec.feature_names == ["cpu"]
    assert task.input_spec.steps_back == 1
    assert task.input_spec.value_range == {"cpu": (0.0, 1.0)}


def test_arima_builder_propagates_model_params_overrides(static_cfg):
    task_cfg = ArimaTaskConfig(
        kind="arima",
        feature="mem",
        model_params={"p": 2, "d": 0, "q": 1},
    )
    task = build_arima_task("mem_forecast_arima", task_cfg, static_cfg)
    assert task.model.default_params == {"p": 2, "d": 0, "q": 1}


def test_xgb_builder_attaches_xgb_prepare_via_loader(prom_cfg):
    task_cfg = XgbTaskConfig(
        kind="xgb",
        feature="cpu",
        value_range=(0.0, 1.0),
        steps_back=6,
        query='avg(rate(node_cpu_seconds_total[30s]))',
    )
    task = build_xgb_task("cpu_forecast_xgb", task_cfg, prom_cfg)
    assert task.model.name == "xgb"
    # The XGB prepare gets passed into the loader; the loader is a
    # PrometheusLoader because source=prometheus.
    assert task.data_loader.__class__.__name__ == "PrometheusLoader"
    assert task.data_loader.value_col == "cpu"
    assert task.data_loader.query == 'avg(rate(node_cpu_seconds_total[30s]))'


def test_lstm_builder_carries_batch_size_into_prepare(prom_cfg):
    task_cfg = LstmTaskConfig(
        kind="lstm",
        feature="cpu",
        steps_back=6,
        batch_size=32,
        query='avg(node_load1)',
        model_params={"hidden_size": 8, "num_epochs": 5},
    )
    task = build_lstm_task("cpu_forecast_lstm", task_cfg, prom_cfg)
    assert task.model.name == "lstm"
    assert task.model.default_params["hidden_size"] == 8
    assert task.model.default_params["num_epochs"] == 5
    assert task.input_spec.steps_back == 6


def test_lstm_builder_propagates_horizon_to_output_size_and_max_horizon(prom_cfg):
    """Wave 1 #2: an LSTM task's ``horizon:`` config field drives both
    the trained ``output_size`` and the ``InputSpec.max_horizon`` clamp.
    """
    from intelligence.config.settings import LstmTaskConfig
    from intelligence.tasks.contracts import InputSpec

    if "horizon" not in LstmTaskConfig.model_fields:
        pytest.skip("LstmTaskConfig.horizon not implemented yet")
    if "max_horizon" not in InputSpec.model_fields:
        pytest.skip("InputSpec.max_horizon not implemented yet")

    task_cfg = LstmTaskConfig(
        kind="lstm",
        feature="cpu",
        steps_back=6,
        batch_size=16,
        horizon=4,
        query="avg(node_load1)",
    )
    task = build_lstm_task("cpu_forecast_lstm", task_cfg, prom_cfg)
    assert task.model.default_params["output_size"] == 4
    assert task.input_spec.max_horizon == 4


def test_arima_builder_leaves_max_horizon_unbounded(static_cfg):
    """ARIMA can serve any horizon — builder must leave ``max_horizon`` ``None``."""
    from intelligence.tasks.contracts import InputSpec

    if "max_horizon" not in InputSpec.model_fields:
        pytest.skip("InputSpec.max_horizon not implemented yet")

    task_cfg = ArimaTaskConfig(kind="arima", feature="cpu")
    task = build_arima_task("cpu_forecast_arima", task_cfg, static_cfg)
    assert task.input_spec.max_horizon is None


def test_xgb_builder_leaves_max_horizon_unbounded(static_cfg):
    """XGB is recursive — builder must leave ``max_horizon`` ``None``."""
    from intelligence.tasks.contracts import InputSpec

    if "max_horizon" not in InputSpec.model_fields:
        pytest.skip("InputSpec.max_horizon not implemented yet")

    task_cfg = XgbTaskConfig(kind="xgb", feature="cpu")
    task = build_xgb_task("cpu_forecast_xgb", task_cfg, static_cfg)
    assert task.input_spec.max_horizon is None


def test_drift_builder_produces_drift_detection_task(prom_cfg):
    task_cfg = DriftTaskConfig(
        kind="drift",
        feature="cpu",
        value_range=(0.0, 1.0),
        forecaster="cpu_forecast_arima",
        chunk_size=12,
        query='avg(rate(node_cpu_seconds_total[30s]))',
    )
    task = build_drift_task("cpu_forecast_arima_drift", task_cfg, prom_cfg)
    assert task.__class__.__name__ == "DriftDetectionTask"
    assert task.forecaster_task_name == "cpu_forecast_arima"
    assert task.chunk_size == 12
    assert task.input_spec.feature_names == ["cpu"]
    assert task.input_spec.steps_back == 12  # drift's steps_back == chunk_size


def test_builders_dict_covers_every_shipped_kind():
    assert set(BUILDERS) == {"arima", "xgb", "lstm", "drift"}


def test_get_builder_raises_for_unknown_kind():
    with pytest.raises(KeyError, match="transformer"):
        get_builder("transformer")


def test_prometheus_query_missing_raises_at_loader_build(prom_cfg):
    """A prom-mode task with no `query:` should fail loudly during
    registry build, not silently at first request."""
    task_cfg = ArimaTaskConfig(kind="arima", feature="cpu", query=None)
    with pytest.raises(ValueError, match="no PromQL query"):
        build_arima_task("cpu_forecast_arima", task_cfg, prom_cfg)
