"""Pydantic request / response schemas for the per-task API surface."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from intelligence.config.settings import DriftMetric


class StaticDataSource(BaseModel):
    """Static data source: read a CSV from the configured samples directory."""

    kind: Literal["static"]
    name: str = Field(..., description="CSV filename in the configured samples directory")


class PrometheusDataSource(BaseModel):
    """PromQL data source.

    ``endpoint`` is an optional per-request override of the configured
    Prometheus URL. Off by default at the service level; the operator
    must set ``intelligence.telemetry.allow_endpoint_override: true``
    for the loader to honour it. SSRF defense: an authenticated POST
    /train can otherwise redirect outbound traffic anywhere.
    """

    kind: Literal["prometheus"]
    window: str = Field(..., description="e.g. '24h'")
    step: str = Field(..., description="e.g. '1m'")
    endpoint: str | None = Field(
        default=None,
        description=(
            "Optional override of the configured Prometheus endpoint. "
            "Honoured only when the service config sets "
            "`telemetry.allow_endpoint_override: true`."
        ),
    )


# Discriminated union — pydantic dispatches on `kind` and returns 422 on unknown values.
DataSource = Annotated[
    StaticDataSource | PrometheusDataSource,
    Field(discriminator="kind"),
]


class TrainRequest(BaseModel):
    data_source: DataSource
    model_parameters: dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    model_tag: str
    metrics: dict[str, Any]


class ForecastPoint(BaseModel):
    """One step of a forecast.

    ``value`` is the point estimate. ``lower`` / ``upper`` bracket a
    95 % confidence interval when the underlying model exposes one
    (ARIMA does, recursive XGB does not, direct-output LSTM does not
    by default). Both bounds present or both absent — never just one.
    """

    value: float
    lower: float | None = None
    upper: float | None = None


class PredictRequest(BaseModel):
    input_series: dict[str, list[float]]
    # Number of steps ahead to forecast. Tasks may bound this via
    # ``InputSpec.max_horizon`` (e.g. an LSTM trained with output_size=N
    # refuses horizon>N at the API boundary).
    horizon: int = Field(1, ge=1, description="Forecast steps ahead (>= 1)")
    # Optional model version pin for this request. Overrides any task-level
    # `pinned_version`. ``None`` falls back to task pin → ``:latest``.
    model_version: str | None = None


class DriftPrediction(BaseModel):
    """Drift task response payload. Mirrors the dict
    ``DriftModel.predict`` returns; declared as a model here so
    ``PredictResponse`` advertises a real shape instead of ``Any``.
    """

    drift_detected: bool
    n_chunks: int
    metric: DriftMetric
    forecaster: str


class PredictResponse(BaseModel):
    # Forecast tasks return ``list[ForecastPoint]`` of length
    # ``request.horizon``; drift tasks return ``DriftPrediction``. The
    # untagged union resolves structurally (list vs object), so the
    # caller doesn't need to know which kind of task served the
    # request — pydantic coerces from the runtime dict either way.
    prediction: list[ForecastPoint] | DriftPrediction
    # The concrete version that actually served this request — useful for
    # logging, A/B analysis, and verifying a rollback took effect.
    model_version: str | None = None


class ModelSyncRequest(BaseModel):
    action: Literal["push", "pull"]
    model_tag: str
    repo_id: str | None = None  # defaults to config.intelligence.model_repo.repo_id
    commit_message: str | None = None  # push only


class ModelSyncResponse(BaseModel):
    action: str
    model_tag: str
    repo_id: str
