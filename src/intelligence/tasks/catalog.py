"""Builtin task factories.

Each factory composes a data loader with a model and registers the
resulting ``BaseTask`` under a name. The factory body is what imports
the model class — modules for unconfigured tasks aren't imported, so
their deps stay opt-in.

Each factory takes the full ``IntelligenceConfig`` and uses
``build_loader_for_task`` to pick its loader — operators flip
``telemetry.source`` once and every task switches between CSV-backed
(dev/test) and PromQL-backed (prod).

Adding ``cpu_forecast_xgb`` later is one new factory:

    @register_builtin("cpu_forecast_xgb")
    def make_cpu_forecast_xgb(cfg: IntelligenceConfig) -> BaseTask:
        from intelligence.ml.models.xgb import XgbModel
        return BaseTask(
            name="cpu_forecast_xgb",
            model=XgbModel(),
            data_loader=build_loader_for_task(cfg, "cpu_forecast_xgb"),
        )

A new task domain (e.g. ``mem_forecast_arima``) is the same line with a
different ``value_col`` argument and per-instance model defaults.
M+N, not M*N.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from intelligence.tasks.base import BaseTask, register_builtin
from intelligence.tasks.loaders import build_loader_for_task

if TYPE_CHECKING:
    from intelligence.config.settings import IntelligenceConfig


@register_builtin("cpu_forecast_arima")
def make_cpu_forecast_arima(cfg: "IntelligenceConfig") -> BaseTask:
    from intelligence.ml.models.arima import ArimaModel
    from intelligence.tasks.contracts import InputSpec
    return BaseTask(
        name="cpu_forecast_arima",
        model=ArimaModel(),
        data_loader=build_loader_for_task(cfg, "cpu_forecast_arima"),
        bento_name="metrics_utilization_model_arima",  # legacy compat
        input_spec=InputSpec(
            n_features=1,
            feature_names=["cpu"],
            steps_back=1,                            # ARIMA forecasts off the last observation
            value_range={"cpu": (0.0, 1.0)},          # fractional CPU like the legacy ICOS data
            units={"cpu": "fraction"},
        ),
    )


@register_builtin("cpu_forecast_xgb")
def make_cpu_forecast_xgb(cfg: "IntelligenceConfig") -> BaseTask:
    from intelligence.ml.models.xgb import XgbModel, make_xgb_prepare
    from intelligence.tasks.contracts import InputSpec

    look_back = 6
    return BaseTask(
        name="cpu_forecast_xgb",
        model=XgbModel(),
        data_loader=build_loader_for_task(
            cfg,
            "cpu_forecast_xgb",
            prepare=make_xgb_prepare(look_back=look_back, num_variables=1),
        ),
        bento_name="metrics_utilization_model_xgb",  # legacy compat
        input_spec=InputSpec(
            n_features=1,
            feature_names=["cpu"],
            steps_back=look_back,
            value_range={"cpu": (0.0, 1.0)},
            units={"cpu": "fraction"},
        ),
    )


@register_builtin("cpu_forecast_arima_drift")
def make_cpu_forecast_arima_drift(cfg: "IntelligenceConfig") -> BaseTask:
    from intelligence.tasks.contracts import InputSpec
    from intelligence.tasks.drift import DriftDetectionTask, make_drift_prepare

    chunk_size = 12
    return DriftDetectionTask(
        name="cpu_forecast_arima_drift",
        forecaster_task_name="cpu_forecast_arima",
        model=None,
        data_loader=build_loader_for_task(
            cfg,
            "cpu_forecast_arima_drift",
            prepare=make_drift_prepare(value_col=None),
        ),
        chunk_size=chunk_size,
        input_spec=InputSpec(
            n_features=1,
            feature_names=["cpu"],
            steps_back=chunk_size,
            value_range={"cpu": (0.0, 1.0)},
            units={"cpu": "fraction"},
        ),
    )


@register_builtin("cpu_forecast_lstm")
def make_cpu_forecast_lstm(cfg: "IntelligenceConfig") -> BaseTask:
    from intelligence.ml.models.lstm import LstmModel, make_lstm_prepare
    from intelligence.tasks.contracts import InputSpec

    look_back = 6
    return BaseTask(
        name="cpu_forecast_lstm",
        model=LstmModel(input_size=1, output_size=1, hidden_size=4, num_epochs=3),
        data_loader=build_loader_for_task(
            cfg,
            "cpu_forecast_lstm",
            prepare=make_lstm_prepare(look_back=look_back, num_variables=1, batch_size=16),
        ),
        bento_name="metrics_utilization_model_lstm",  # legacy compat
        input_spec=InputSpec(
            n_features=1,
            feature_names=["cpu"],
            steps_back=look_back,
            value_range={"cpu": (0.0, 1.0)},
            units={"cpu": "fraction"},
        ),
    )
