"""Builtin task factories.

Each factory composes a data loader with a model adapter and registers
the resulting ``BaseTask`` under a name. The factory body is what
imports the model adapter — modules for unconfigured tasks aren't
imported, so their deps stay opt-in.

Adding ``cpu_forecast_xgb`` later is one new factory:

    @register_builtin("cpu_forecast_xgb")
    def make_cpu_forecast_xgb() -> BaseTask:
        from intelligence.models.xgb import XgbAdapter
        return BaseTask(
            name="cpu_forecast_xgb",
            model_adapter=XgbAdapter(),
            data_loader=static_csv_loader(),
        )

A new task domain (e.g. ``mem_forecast_arima``) is the same line with a
different ``data_loader`` and per-instance adapter defaults. M+N, not M×N.
"""

from __future__ import annotations

from intelligence.tasks.base import BaseTask, register_builtin
from intelligence.tasks.loaders import static_csv_loader


@register_builtin("cpu_forecast_arima")
def make_cpu_forecast_arima() -> BaseTask:
    from intelligence.models.arima import ArimaAdapter
    return BaseTask(
        name="cpu_forecast_arima",
        model_adapter=ArimaAdapter(),
        data_loader=static_csv_loader(),
        bento_name="metrics_utilization_model_arima",  # legacy compat
    )
