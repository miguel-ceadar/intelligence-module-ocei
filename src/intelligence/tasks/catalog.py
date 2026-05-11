"""Builtin task factories.

Each factory composes a data loader with a model adapter and registers
the resulting ``ForecastTask`` under a name. The factory body is what
imports the model adapter — modules for unconfigured tasks aren't
imported, so their deps stay opt-in.

Adding ``cpu_forecast_xgb`` later is one new factory:

    @register_builtin("cpu_forecast_xgb")
    def make_cpu_forecast_xgb() -> ForecastTask:
        from intelligence.models.xgb import XgbAdapter
        return ForecastTask(
            name="cpu_forecast_xgb",
            model_adapter=XgbAdapter(),
            data_loader=static_csv_loader(),
            bento_name="metrics_utilization_model_xgb",
        )

A new task domain (e.g. ``mem_forecast_arima``) is the same line with a
different ``data_loader`` and ``bento_name``. M+N, not M×N.
"""

from __future__ import annotations

from intelligence.tasks.base import register_builtin
from intelligence.tasks.forecast import ForecastTask
from intelligence.tasks.loaders import static_csv_loader


@register_builtin("cpu_forecast_arima")
def make_cpu_forecast_arima() -> ForecastTask:
    from intelligence.models.arima import ArimaAdapter
    return ForecastTask(
        name="cpu_forecast_arima",
        model_adapter=ArimaAdapter(),
        data_loader=static_csv_loader(),
        bento_name="metrics_utilization_model_arima",  # legacy compat
    )
