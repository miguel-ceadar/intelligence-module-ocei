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
from pathlib import Path
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

    # ---- New manifest-based protocol --------------------------------------
    #
    # ``fit`` / ``save_artifacts`` / ``load_artifacts`` all speak the same
    # ``artifacts`` dict shape — what predict needs at runtime. The legacy
    # ``train`` / ``predict(bento, ...)`` methods below stay until step 8
    # flips ``BaseTask`` to call the new path; step 9 then removes them.

    def fit(self, components: dict) -> tuple[dict, dict]:
        """Train and return ``(artifacts, metrics)``.

        ``artifacts`` carries the runtime state predict consumes: scaler,
        full history series (after walk-forward), the ARIMA order, and
        the test sample size. ``BaseTask`` injects ``input_spec`` into
        this dict before calling ``save_artifacts``.
        """
        from intelligence.ml.trainers import ModelTrainer

        order_params = {**self.default_params, **components.get("model_parameters", {})}
        components_with_params = {**components, "model_parameters": order_params}

        trainer = ModelTrainer(components_with_params)
        metrics, _model, history, _y_test, _y_pred = trainer.train_arima()
        metrics_jsonable = _coerce_jsonable(metrics)

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
        """Persist the artefacts as a flat directory and return the
        ``role -> filename`` map for the manifest.

        ARIMA itself has no native serialisable model — we re-fit on
        every predict call against the persisted history — so the
        on-disk model file is just ``arima.json`` (order + history).
        """
        from intelligence.ml.artifact.sidecars import (
            save_input_spec,
            save_json,
            save_sklearn_scaler,
        )

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
        """Inverse of :meth:`save_artifacts` — returns the same dict
        shape that :meth:`fit` emits, plus ``input_spec`` if persisted.
        """
        from intelligence.ml.artifact.sidecars import (
            load_input_spec,
            load_json,
            load_sklearn_scaler,
        )

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
