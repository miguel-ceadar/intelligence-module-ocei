"""ARIMA model adapter.

Wraps ``ModelTrainer.train_arima`` for the ``ModelAdapter`` contract.
The Bento layout (``custom_objects: scaler_obj, historical_data,
model_metrics, test_sample_size``) matches what the legacy
``oasis/api_service.py`` predict branch expects, so saved Bentos can be
served through either path during phase 1.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ArimaAdapter:
    """ARIMA model adapter.

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
        from intelligence.trainers import ModelTrainer

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

    def predict(self, bento_model: Any, input_series: dict[str, list[float]]) -> Any:
        # ARIMA is univariate — pick the first input series.
        if not input_series:
            raise ValueError("input_series is empty")
        _key, values = next(iter(input_series.items()))
        if not values:
            raise ValueError("input_series values are empty")

        scaler = bento_model.custom_objects["scaler_obj"]
        history = list(bento_model.custom_objects.get("historical_data", []))
        order = tuple(bento_model.custom_objects.get(
            "arima_order",
            (self.default_params["p"], self.default_params["d"], self.default_params["q"]),
        ))

        from statsmodels.tsa.arima.model import ARIMA

        last_scaled = float(scaler.transform(np.array([[values[-1]]]))[0][0])
        history.append(last_scaled)
        fit = ARIMA(history, order=order).fit()
        yhat_scaled = float(fit.forecast()[0])
        yhat = float(scaler.inverse_transform(np.array([[yhat_scaled]]))[0][0])
        return round(yhat, 4)


def _coerce_jsonable(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        out[k] = v.item() if hasattr(v, "item") else v
    return out
