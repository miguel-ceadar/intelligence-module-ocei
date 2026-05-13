"""ARIMA model.

Wraps ``ModelTrainer.train_arima`` for the ``Model`` contract. The
Bento's ``custom_objects`` carry the fitted scaler, the historical
training series, and per-task metrics so ``predict`` can refit
against the stored prior plus the new observation.

Multi-horizon forecasts come straight from statsmodels'
``get_forecast(steps=N).summary_frame()``, which also exposes the 95 %
confidence band — populated into each ``ForecastPoint.lower`` /
``upper``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from intelligence.api.schemas import ForecastPoint

logger = logging.getLogger(__name__)


class ArimaModel:
    """ARIMA model.

    Defaults are *per-instance*: two registered tasks both using ARIMA
    can carry different baseline orders (e.g. ``cpu_forecast_arima``
    with ``p=5`` and ``mem_forecast_arima`` with ``p=3``) without
    subclassing or per-request overrides.
    """

    name = "arima"
    has_drift = True

    def __init__(self, p: int = 5, d: int = 1, q: int = 0) -> None:
        self.default_params = {"p": p, "d": d, "q": q}

    def train(
        self,
        components: dict,
        bento_name: str,
        extras: dict | None = None,
    ) -> tuple[Any, dict]:
        from intelligence.ml.trainers import ModelTrainer

        order_params = {**self.default_params, **components.get("model_parameters", {})}
        components_with_params = {**components, "model_parameters": order_params}

        trainer = ModelTrainer(components_with_params)
        metrics, model, history, _y_test, _y_pred = trainer.train_arima()

        custom_objects = {
            "scaler_obj": components_with_params["scaler_obj"],
            "historical_data": history,
            "model_metrics": metrics,
            "test_sample_size": len(components_with_params["X_test"]),
            "arima_order": (order_params["p"], order_params["d"], order_params["q"]),
            **(extras or {}),
        }

        import bentoml

        bento = bentoml.picklable_model.save_model(
            bento_name,
            model,
            custom_objects=custom_objects,
            signatures={"predict": {"batchable": True}},
        )
        return bento, _coerce_jsonable(metrics)

    def predict(
        self,
        bento_model: Any,
        input_series: dict[str, list[float]],
        horizon: int = 1,
    ) -> list[ForecastPoint]:
        # ARIMA is univariate — pick the first input series.
        if not input_series:
            raise ValueError("input_series is empty")
        _key, values = next(iter(input_series.items()))
        if not values:
            raise ValueError("input_series values are empty")

        scaler = bento_model.custom_objects["scaler_obj"]
        history = list(bento_model.custom_objects.get("historical_data", []))
        order = tuple(
            bento_model.custom_objects.get(
                "arima_order",
                (self.default_params["p"], self.default_params["d"], self.default_params["q"]),
            )
        )

        from statsmodels.tsa.arima.model import ARIMA

        last_scaled = float(scaler.transform(np.array([[values[-1]]]))[0][0])
        history.append(last_scaled)
        fit = ARIMA(history, order=order).fit()

        # 95 % CI is what statsmodels returns by default (alpha=0.05).
        frame = fit.get_forecast(steps=horizon).summary_frame(alpha=0.05)
        mean_raw = scaler.inverse_transform(frame["mean"].to_numpy().reshape(-1, 1)).flatten()
        lower_raw = scaler.inverse_transform(
            frame["mean_ci_lower"].to_numpy().reshape(-1, 1)
        ).flatten()
        upper_raw = scaler.inverse_transform(
            frame["mean_ci_upper"].to_numpy().reshape(-1, 1)
        ).flatten()

        return [
            ForecastPoint(
                value=round(float(m), 4),
                lower=round(float(lo), 4),
                upper=round(float(hi), 4),
            )
            for m, lo, hi in zip(mean_raw, lower_raw, upper_raw, strict=True)
        ]


def _coerce_jsonable(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        out[k] = v.item() if hasattr(v, "item") else v
    return out
