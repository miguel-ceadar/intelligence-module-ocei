"""Tests for per-kind task builders.

Verifies that each builder produces a ``BaseTask`` instance whose
public surface matches what today's hand-rolled factories produce —
same task name, same InputSpec, same model class/params.
"""

from __future__ import annotations

import pytest

from intelligence.config.settings import (
    ArimaTaskConfig,
    DriftTaskConfig,
    FeatureSpec,
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
        features=[FeatureSpec(name="cpu", value_range=(0.0, 1.0))],
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
        features=[FeatureSpec(name="mem")],
        model_params={"p": 2, "d": 0, "q": 1},
    )
    task = build_arima_task("mem_forecast_arima", task_cfg, static_cfg)
    assert task.model.default_params == {"p": 2, "d": 0, "q": 1}


def test_xgb_builder_attaches_xgb_prepare_via_loader(prom_cfg):
    task_cfg = XgbTaskConfig(
        kind="xgb",
        features=[
            FeatureSpec(
                name="cpu",
                value_range=(0.0, 1.0),
                query="avg(rate(node_cpu_seconds_total[30s]))",
            ),
        ],
        steps_back=6,
    )
    task = build_xgb_task("cpu_forecast_xgb", task_cfg, prom_cfg)
    assert task.model.name == "xgb"
    # The XGB prepare gets passed into the loader; the loader is a
    # PrometheusLoader because source=prometheus.
    assert task.data_loader.__class__.__name__ == "PrometheusLoader"
    assert task.data_loader.value_cols == ["cpu"]
    assert task.data_loader.queries == ["avg(rate(node_cpu_seconds_total[30s]))"]


def test_lstm_builder_carries_batch_size_into_prepare(prom_cfg):
    task_cfg = LstmTaskConfig(
        kind="lstm",
        features=[FeatureSpec(name="cpu", query="avg(node_load1)")],
        steps_back=6,
        batch_size=32,
        model_params={"hidden_size": 8, "num_epochs": 5},
    )
    task = build_lstm_task("cpu_forecast_lstm", task_cfg, prom_cfg)
    assert task.model.name == "lstm"
    assert task.model.default_params["hidden_size"] == 8
    assert task.model.default_params["num_epochs"] == 5
    assert task.input_spec.steps_back == 6


def test_lstm_builder_propagates_horizon_to_output_size_and_max_horizon(prom_cfg):
    """An LSTM task's ``horizon:`` config field drives both the trained
    ``output_size`` and the ``InputSpec.max_horizon`` clamp.
    """
    from intelligence.config.settings import LstmTaskConfig
    from intelligence.tasks.contracts import InputSpec

    if "horizon" not in LstmTaskConfig.model_fields:
        pytest.skip("LstmTaskConfig.horizon not implemented yet")
    if "max_horizon" not in InputSpec.model_fields:
        pytest.skip("InputSpec.max_horizon not implemented yet")

    task_cfg = LstmTaskConfig(
        kind="lstm",
        features=[FeatureSpec(name="cpu", query="avg(node_load1)")],
        steps_back=6,
        batch_size=16,
        horizon=4,
    )
    task = build_lstm_task("cpu_forecast_lstm", task_cfg, prom_cfg)
    assert task.model.default_params["output_size"] == 4
    assert task.input_spec.max_horizon == 4


def test_arima_builder_leaves_max_horizon_unbounded(static_cfg):
    """ARIMA can serve any horizon — builder must leave ``max_horizon`` ``None``."""
    from intelligence.tasks.contracts import InputSpec

    if "max_horizon" not in InputSpec.model_fields:
        pytest.skip("InputSpec.max_horizon not implemented yet")

    task_cfg = ArimaTaskConfig(kind="arima", features=[FeatureSpec(name="cpu")])
    task = build_arima_task("cpu_forecast_arima", task_cfg, static_cfg)
    assert task.input_spec.max_horizon is None


def test_xgb_builder_leaves_max_horizon_unbounded(static_cfg):
    """XGB is recursive — builder must leave ``max_horizon`` ``None``."""
    from intelligence.tasks.contracts import InputSpec

    if "max_horizon" not in InputSpec.model_fields:
        pytest.skip("InputSpec.max_horizon not implemented yet")

    task_cfg = XgbTaskConfig(kind="xgb", features=[FeatureSpec(name="cpu")])
    task = build_xgb_task("cpu_forecast_xgb", task_cfg, static_cfg)
    assert task.input_spec.max_horizon is None


def test_drift_builder_produces_basetask_with_drift_model(prom_cfg):
    """A drift task is a plain ``BaseTask`` wired with a ``DriftModel``;
    drift-specific config lives on the wrapped model. ``chunk_size``
    doubles as ``steps_back`` on the InputSpec so the contract layer
    enforces exact-length analysis windows.
    """
    from intelligence.ml.models.drift import DriftModel
    from intelligence.tasks.base import BaseTask

    task_cfg = DriftTaskConfig(
        kind="drift",
        features=[
            FeatureSpec(
                name="cpu",
                value_range=(0.0, 1.0),
                query="avg(rate(node_cpu_seconds_total[30s]))",
            ),
        ],
        forecaster="cpu_forecast_arima",
        chunk_size=12,
    )
    task = build_drift_task("cpu_forecast_arima_drift", task_cfg, prom_cfg)
    assert isinstance(task, BaseTask)
    assert isinstance(task.model, DriftModel)
    assert task.model.forecaster_task_name == "cpu_forecast_arima"
    assert task.model.chunk_size == 12
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
    task_cfg = ArimaTaskConfig(kind="arima", features=[FeatureSpec(name="cpu")])
    with pytest.raises(ValueError, match="no PromQL query"):
        build_arima_task("cpu_forecast_arima", task_cfg, prom_cfg)


def test_arima_builder_rejects_multivariate_loudly(static_cfg):
    """ARIMA is univariate by construction. Multivariate would need VAR
    (a different statsmodels API), shipping later as `kind: var`. The
    builder must refuse the config at registry build time so the
    misconfig surfaces at startup rather than at first /train."""
    task_cfg = ArimaTaskConfig(
        kind="arima",
        features=[FeatureSpec(name="cpu"), FeatureSpec(name="memory")],
    )
    with pytest.raises(ValueError, match=r"ARIMA.*univariate"):
        build_arima_task("cpu_plus_mem_arima", task_cfg, static_cfg)


def test_xgb_builder_propagates_num_variables_to_prepare(prom_cfg):
    """XGB builder passes ``num_variables=len(features)`` so the prepare
    sees the full feature set. Training the multivariate path is Gate 4
    work — this only verifies the builder composes."""
    task_cfg = XgbTaskConfig(
        kind="xgb",
        features=[
            FeatureSpec(name="cpu", query="avg(rate(node_cpu[30s]))", value_range=(0.0, 1.0)),
            FeatureSpec(name="memory", query="avg(node_mem_avail_bytes)"),
        ],
        steps_back=6,
    )
    task = build_xgb_task("cpu_plus_mem_xgb", task_cfg, prom_cfg)
    assert task.input_spec.n_features == 2
    assert task.input_spec.feature_names == ["cpu", "memory"]
    assert task.data_loader.value_cols == ["cpu", "memory"]
    assert task.data_loader.queries == [
        "avg(rate(node_cpu[30s]))",
        "avg(node_mem_avail_bytes)",
    ]


def test_lstm_builder_propagates_num_variables_to_prepare(prom_cfg):
    """LSTM builder passes ``num_variables=len(features)``. The prepare
    already shapes ``(samples, look_back, num_variables)`` tensors;
    Gate 4 handles output_size for target-vs-multi-output semantics."""
    task_cfg = LstmTaskConfig(
        kind="lstm",
        features=[
            FeatureSpec(name="cpu", query="avg(node_cpu_util)"),
            FeatureSpec(name="memory", query="avg(node_mem_util)"),
            FeatureSpec(name="load", query="avg(node_load1)"),
        ],
        steps_back=6,
    )
    task = build_lstm_task("triple_input_lstm", task_cfg, prom_cfg)
    assert task.input_spec.n_features == 3
    assert task.input_spec.feature_names == ["cpu", "memory", "load"]
    assert task.data_loader.value_cols == ["cpu", "memory", "load"]


def test_lstm_builder_sets_input_size_to_feature_count(prom_cfg):
    """The trained network's ``input_size`` must equal ``len(features)``
    or PyTorch raises ``input.size(-1) must be equal to input_size``
    on the first batch. The builder overrides whatever default the
    task's ``model_params`` carries — input_size is a function of the
    feature count, not a tunable.
    """
    task_cfg = LstmTaskConfig(
        kind="lstm",
        features=[
            FeatureSpec(name="cpu", query="avg(node_cpu_util)"),
            FeatureSpec(name="memory", query="avg(node_mem_util)"),
        ],
        steps_back=6,
        # Operator deliberately set the wrong input_size; the builder
        # must overwrite it.
        model_params={"input_size": 1, "hidden_size": 4, "num_epochs": 2},
    )
    task = build_lstm_task("cpu_mem_lstm", task_cfg, prom_cfg)
    assert task.model.default_params["input_size"] == 2
