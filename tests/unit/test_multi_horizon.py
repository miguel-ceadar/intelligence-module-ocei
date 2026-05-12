"""Wave 1 #2 — multi-horizon forecasting + confidence intervals.

Encodes the contract for the new API shape before any model adapter changes:

  - ``ForecastPoint`` (value, lower?, upper?) replaces the scalar ``prediction``.
  - ``PredictRequest.horizon: int = 1`` (validated ``>= 1``).
  - ``PredictResponse.prediction`` is a list of ``ForecastPoint`` of length
    ``horizon``. Drift tasks keep a dict-shaped prediction — that path is
    unchanged.
  - ``InputSpec.max_horizon: int | None`` lets a task bound the request horizon
    (LSTM direct multi-output sets it to ``output_size``; ARIMA / XGB-recursive
    leave it ``None`` = unbounded).

Tests skip cleanly when a not-yet-shipped field is missing, so the suite stays
green-with-skips during the journey.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intelligence.api import schemas as api_schemas
from intelligence.tasks.contracts import InputSpec


def _maybe(module, name: str):
    obj = getattr(module, name, None)
    if obj is None:
        pytest.skip(f"{module.__name__}.{name} not implemented yet")
    return obj


# --- ForecastPoint ----------------------------------------------------------


def test_forecast_point_schema_is_exported():
    ForecastPoint = _maybe(api_schemas, "ForecastPoint")
    point = ForecastPoint(value=0.42)
    assert point.value == pytest.approx(0.42)
    assert point.lower is None
    assert point.upper is None


def test_forecast_point_carries_optional_interval():
    ForecastPoint = _maybe(api_schemas, "ForecastPoint")
    point = ForecastPoint(value=0.42, lower=0.30, upper=0.55)
    assert point.lower == pytest.approx(0.30)
    assert point.upper == pytest.approx(0.55)


# --- PredictRequest.horizon -------------------------------------------------


def test_predict_request_horizon_defaults_to_one():
    req = api_schemas.PredictRequest(input_series={"cpu": [0.5]})
    if not hasattr(req, "horizon"):
        pytest.skip("PredictRequest.horizon not implemented yet")
    assert req.horizon == 1


def test_predict_request_accepts_explicit_horizon():
    if "horizon" not in api_schemas.PredictRequest.model_fields:
        pytest.skip("PredictRequest.horizon not implemented yet")
    req = api_schemas.PredictRequest(input_series={"cpu": [0.5]}, horizon=12)
    assert req.horizon == 12


def test_predict_request_rejects_zero_or_negative_horizon():
    if "horizon" not in api_schemas.PredictRequest.model_fields:
        pytest.skip("PredictRequest.horizon not implemented yet")
    with pytest.raises(ValidationError):
        api_schemas.PredictRequest(input_series={"cpu": [0.5]}, horizon=0)
    with pytest.raises(ValidationError):
        api_schemas.PredictRequest(input_series={"cpu": [0.5]}, horizon=-3)


# --- PredictResponse list shape ---------------------------------------------


def test_predict_response_carries_list_of_forecast_points():
    ForecastPoint = _maybe(api_schemas, "ForecastPoint")
    resp = api_schemas.PredictResponse(
        prediction=[ForecastPoint(value=0.4), ForecastPoint(value=0.45, lower=0.4, upper=0.5)],
        model_version="abc123",
    )
    # round-trip serialization carries the nested fields.
    payload = resp.model_dump()
    assert isinstance(payload["prediction"], list)
    assert payload["prediction"][0]["value"] == pytest.approx(0.4)
    assert payload["prediction"][1]["lower"] == pytest.approx(0.4)
    assert payload["prediction"][1]["upper"] == pytest.approx(0.5)


# --- InputSpec.max_horizon --------------------------------------------------


def test_input_spec_max_horizon_defaults_to_none():
    if "max_horizon" not in InputSpec.model_fields:
        pytest.skip("InputSpec.max_horizon not implemented yet")
    spec = InputSpec(n_features=1, feature_names=["cpu"], steps_back=6)
    assert spec.max_horizon is None


def test_input_spec_max_horizon_round_trips():
    if "max_horizon" not in InputSpec.model_fields:
        pytest.skip("InputSpec.max_horizon not implemented yet")
    spec = InputSpec(n_features=1, feature_names=["cpu"], steps_back=6, max_horizon=4)
    assert spec.max_horizon == 4
