"""Multi-horizon forecasting + confidence intervals.

Pins the API shape:

  - ``ForecastPoint`` (value, lower?, upper?) replaces the scalar ``prediction``.
  - ``PredictRequest.horizon: int = 1`` (validated ``>= 1``).
  - ``PredictResponse.prediction`` is ``list[ForecastPoint] | DriftPrediction``;
    forecast tasks return the list, drift tasks return the model. The untagged
    union resolves structurally (list vs object).
  - ``InputSpec.max_horizon: int | None`` lets a task bound the request
    horizon (LSTM direct multi-output sets it to ``output_size``; ARIMA /
    XGB-recursive leave it ``None`` = unbounded).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intelligence.api.schemas import DriftPrediction, ForecastPoint, PredictRequest, PredictResponse
from intelligence.tasks.contracts import InputSpec

# --- ForecastPoint ----------------------------------------------------------


def test_forecast_point_schema_is_exported():
    point = ForecastPoint(value=0.42)
    assert point.value == pytest.approx(0.42)
    assert point.lower is None
    assert point.upper is None


def test_forecast_point_carries_optional_interval():
    point = ForecastPoint(value=0.42, lower=0.30, upper=0.55)
    assert point.lower == pytest.approx(0.30)
    assert point.upper == pytest.approx(0.55)


# --- PredictRequest.horizon -------------------------------------------------


def test_predict_request_horizon_defaults_to_one():
    req = PredictRequest(input_series={"cpu": [0.5]})
    assert req.horizon == 1


def test_predict_request_accepts_explicit_horizon():
    req = PredictRequest(input_series={"cpu": [0.5]}, horizon=12)
    assert req.horizon == 12


def test_predict_request_rejects_zero_or_negative_horizon():
    with pytest.raises(ValidationError):
        PredictRequest(input_series={"cpu": [0.5]}, horizon=0)
    with pytest.raises(ValidationError):
        PredictRequest(input_series={"cpu": [0.5]}, horizon=-3)


# --- PredictResponse list shape ---------------------------------------------


def test_predict_response_carries_list_of_forecast_points():
    resp = PredictResponse(
        prediction=[ForecastPoint(value=0.4), ForecastPoint(value=0.45, lower=0.4, upper=0.5)],
        model_version="abc123",
    )
    # round-trip serialization carries the nested fields.
    payload = resp.model_dump()
    assert isinstance(payload["prediction"], list)
    assert payload["prediction"][0]["value"] == pytest.approx(0.4)
    assert payload["prediction"][1]["lower"] == pytest.approx(0.4)
    assert payload["prediction"][1]["upper"] == pytest.approx(0.5)


# --- DriftPrediction + PredictResponse union --------------------------------


def test_drift_prediction_schema_matches_runtime_dict():
    """``DriftModel.predict`` returns a dict with these four keys. The
    schema has to match the runtime contract so untagged-union coercion
    works without the caller having to instantiate the model itself.
    """
    pred = DriftPrediction(
        drift_detected=True,
        n_chunks=3,
        metric="jensen_shannon",
        forecaster="cpu_forecast_arima",
    )
    assert pred.drift_detected is True
    assert pred.n_chunks == 3


def test_predict_response_coerces_drift_dict_into_drift_prediction():
    """``BaseTask.predict`` builds ``PredictResponse(prediction=<drift dict>)``
    — pydantic must coerce the dict to ``DriftPrediction`` via the union.
    """
    resp = PredictResponse(
        prediction={
            "drift_detected": False,
            "n_chunks": 1,
            "metric": "jensen_shannon",
            "forecaster": "cpu_forecast_arima",
        },
        model_version="abc",
    )
    assert isinstance(resp.prediction, DriftPrediction)
    assert resp.prediction.drift_detected is False


def test_predict_response_rejects_unrelated_dict():
    """An object that's neither a list nor a drift shape should fail —
    the old ``Any`` schema accepted anything."""
    with pytest.raises(ValidationError):
        PredictResponse(prediction={"unrelated": "data"}, model_version="x")


def test_openapi_advertises_real_response_models():
    """``response_model=`` on the FastAPI handlers should surface
    ``PredictResponse`` and ``TrainResponse`` in the OpenAPI schema —
    consumers can generate clients against the real types instead of
    inferring shapes by trial-and-error."""
    from intelligence.api import service as api_service

    schema = api_service.app.openapi()
    predict = schema["paths"]["/tasks/{task_name}/predict"]["post"]
    train = schema["paths"]["/tasks/{task_name}/train"]["post"]
    predict_ref = predict["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    train_ref = train["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert predict_ref.endswith("/PredictResponse")
    assert train_ref.endswith("/TrainResponse")


# --- InputSpec.max_horizon --------------------------------------------------


def test_input_spec_max_horizon_defaults_to_none():
    spec = InputSpec(n_features=1, feature_names=["cpu"], steps_back=6)
    assert spec.max_horizon is None


def test_input_spec_max_horizon_round_trips():
    spec = InputSpec(n_features=1, feature_names=["cpu"], steps_back=6, max_horizon=4)
    assert spec.max_horizon == 4
