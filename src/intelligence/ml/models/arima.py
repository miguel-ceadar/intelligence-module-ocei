"""ARIMA model implementing the ``Model`` protocol.

Predict refits the ARIMA against the persisted history plus the new
observation. Multi-horizon forecasts come from statsmodels'
``get_forecast(steps=N).summary_frame()``, which also provides the
95 % confidence band populated into each ``ForecastPoint``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from intelligence.api.schemas import ForecastPoint
from intelligence.ml.artifact.sidecars import (
    load_input_spec,
    load_json,
    load_sklearn_scaler,
    save_input_spec,
    save_json,
    save_sklearn_scaler,
)
from intelligence.ml.models._common import coerce_metrics
from intelligence.ml.trainers import ModelTrainer

logger = logging.getLogger(__name__)


class ArimaModel:
    """ARIMA forecaster.

    Defaults are per-instance, so two registered tasks both using
    ARIMA can carry different baseline orders without subclassing.
    """

    name = "arima"
    has_drift = True

    def __init__(self, p: int = 5, d: int = 1, q: int = 0) -> None:
        self.default_params = {"p": p, "d": d, "q": q}

    def fit(self, components: dict) -> tuple[dict, dict]:
        """Train and return ``(artifacts, metrics)``. ``artifacts``
        carries the scaler, the walk-forward history series, the
        ARIMA order, and the test sample size.
        """
        order_params = {**self.default_params, **components.get("model_parameters", {})}
        components_with_params = {**components, "model_parameters": order_params}

        trainer = ModelTrainer(components_with_params)
        metrics, history = trainer.train_arima()
        metrics_jsonable = coerce_metrics(metrics)

        artifacts = {
            "scaler_obj": components_with_params["scaler_obj"],
            "historical_data": list(history),
            "arima_order": (
                order_params["p"],
                order_params["d"],
                order_params["q"],
            ),
            "model_metrics": metrics_jsonable,
            "test_sample_size": len(components_with_params["X_test"]),
        }
        return artifacts, metrics_jsonable

    def save_artifacts(self, artifacts: dict, dest: Path) -> dict[str, str]:
        """Persist the artifacts and return the ``role -> filename`` map.

        ARIMA refits on every predict call, so the on-disk model file
        is just ``arima.json`` (order + history).
        """
        save_json(
            dest,
            "arima.json",
            {
                "order": list(artifacts["arima_order"]),
                "history": list(artifacts["historical_data"]),
                "test_sample_size": int(artifacts.get("test_sample_size", 0)),
            },
        )
        save_sklearn_scaler(dest, "scaler", artifacts["scaler_obj"])
        save_json(dest, "metrics.json", artifacts.get("model_metrics", {}))

        files: dict[str, str] = {
            "model": "arima.json",
            "scaler_meta": "scaler.json",
            "scaler_arrays": "scaler.npz",
            "metrics": "metrics.json",
        }

        spec = artifacts.get("input_spec")
        if spec is not None:
            save_input_spec(dest, spec)
            files["input_spec"] = "input_spec.json"

        return files

    def load_artifacts(self, src: Path) -> dict:
        """Restore the dict shape ``fit`` emits, plus ``input_spec``
        if it was persisted.
        """
        arima_data = load_json(src, "arima.json")
        loaded: dict[str, Any] = {
            "scaler_obj": load_sklearn_scaler(src, "scaler"),
            "historical_data": list(arima_data["history"]),
            "arima_order": tuple(arima_data["order"]),
            "model_metrics": load_json(src, "metrics.json"),
            "test_sample_size": int(arima_data.get("test_sample_size", 0)),
        }
        if (src / "input_spec.json").exists():
            loaded["input_spec"] = load_input_spec(src)
        return loaded

    def predict(
        self,
        artifacts: dict,
        input_series: dict[str, list[float]],
        horizon: int = 1,
    ) -> list[ForecastPoint]:
        # ARIMA is univariate — pick the first input series.
        if not input_series:
            raise ValueError("input_series is empty")
        _key, values = next(iter(input_series.items()))
        if not values:
            raise ValueError("input_series values are empty")

        # ``scaler_obj`` is the y-scaler in every model. ARIMA is the
        # only one that also uses it to *scale the input* below —
        # that's safe because ARIMA is univariate, so the input series
        # *is* the target. Adding covariates would require a separate
        # input scaler (cf. XGB's ``scaler_X``).
        scaler = artifacts["scaler_obj"]
        history = list(artifacts.get("historical_data", []))
        order = tuple(
            artifacts.get(
                "arima_order",
                (
                    self.default_params["p"],
                    self.default_params["d"],
                    self.default_params["q"],
                ),
            )
        )

        last_scaled = float(scaler.transform(np.array([[values[-1]]]))[0][0])
        history.append(last_scaled)
        fit = ARIMA(history, order=order).fit()

        # alpha=0.05 → 95 % CI.
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
