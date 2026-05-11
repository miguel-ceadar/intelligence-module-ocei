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
    name = "arima"
    has_drift = True

    DEFAULT_PARAMS = {"p": 5, "d": 1, "q": 0}

    def train(self, components: dict, bento_name: str) -> tuple[Any, dict]:
        from intelligence.trainers import ModelTrainer

        order_params = {**self.DEFAULT_PARAMS, **components.get("model_parameters", {})}
        components_with_params = {**components, "model_parameters": order_params}

        trainer = ModelTrainer(components_with_params)
        metrics, model, history, _y_test, _y_pred = trainer.train_arima()

        import bentoml
        bento = bentoml.picklable_model.save_model(
            bento_name,
            model,
            custom_objects={
                "scaler_obj": components_with_params["scaler_obj"],
                "historical_data": history,
                "model_metrics": metrics,
                "test_sample_size": len(components_with_params["X_test"]),
                "arima_order": (order_params["p"], order_params["d"], order_params["q"]),
            },
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
            (self.DEFAULT_PARAMS["p"], self.DEFAULT_PARAMS["d"], self.DEFAULT_PARAMS["q"]),
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
