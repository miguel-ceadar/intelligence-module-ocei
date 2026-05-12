"""Pydantic request / response schemas for the per-task API surface."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class StaticDataSource(BaseModel):
    """Static data source: read a CSV from the configured samples directory."""

    kind: Literal["static"]
    name: str = Field(..., description="CSV filename in the configured samples directory")


class PrometheusDataSource(BaseModel):
    """Phase-2 data source: PromQL window. Phase 1 returns 501 for this."""

    kind: Literal["prometheus"]
    window: str = Field(..., description="e.g. '24h'")
    step: str = Field(..., description="e.g. '1m'")


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


class PredictRequest(BaseModel):
    input_series: dict[str, list[float]]


class PredictResponse(BaseModel):
    prediction: Any
    metric_type: int | None = None


class TaskInfo(BaseModel):
    name: str
    model_type: str
    has_drift: bool


class ModelSyncRequest(BaseModel):
    action: Literal["push", "pull"]
    model_tag: str
    repo_id: str | None = None    # defaults to config.intelligence.model_repo.repo_id
    commit_message: str | None = None  # push only


class ModelSyncResponse(BaseModel):
    action: str
    model_tag: str
    repo_id: str
