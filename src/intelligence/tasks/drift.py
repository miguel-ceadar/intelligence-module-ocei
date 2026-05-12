"""Drift detection — a ``BaseTask`` subclass that fits a NannyML
``UnivariateDriftCalculator`` on reference data and flags drift on
fresh chunks.

Scope (phase 2): **input-data drift**. The calculator is fit on the
feature distribution at train time. At predict time, it's applied to
an input chunk; ``prediction = {"drift_detected": bool, ...}``.

The ``forecaster_task_name`` field carries the identity of the
forecaster this drift task is paired with. It's used in the registered
task name and stored in the Bento so an operator can trace the link.
At this stage we do not load the forecaster's Bento — so drift task
training is independent of whether the forecaster has been trained.
A future extension (prediction-drift) can pull the forecaster Bento
at train time to score the reference data first.

The prepare callable must yield ``{"reference_df": pd.DataFrame,
"drift_columns": list[str]}``. Use ``make_drift_prepare`` for the
default univariate shape.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from intelligence.api.schemas import (
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from intelligence.tasks.base import BaseTask

logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionTask(BaseTask):
    """``BaseTask`` subclass: NannyML-backed input-data drift detection.

    Inherits the lazy-load + readiness machinery from ``BaseTask``.
    Overrides ``train`` and ``predict`` because the lifecycle is
    distinct (no per-algorithm Model, no scaler — the calculator is
    the artifact).
    """

    # Drift fields. Dataclass field-ordering rule: every field after a
    # parent's non-default field must itself have a default. The parent
    # ``BaseTask.model`` is the last non-default; everything below has
    # defaults or matches that pattern.
    forecaster_task_name: str = ""
    chunk_size: int = 6
    metric: str = "jensen_shannon"

    def train(self, req: TrainRequest) -> TrainResponse:
        components = self.data_loader(req.data_source)
        reference_df: pd.DataFrame = components["reference_df"]
        column_names: list[str] = list(
            components.get("drift_columns")
            or [c for c in reference_df.columns if c.lower() not in {"time", "timestamp", "date"}]
        )

        try:
            import nannyml as nml
        except ImportError as e:
            raise ImportError(
                "drift task requires the `nannyml` package. Install it with "
                "`uv sync --extra drift` (or `pip install intelligence[drift]`) "
                "and rebuild the image."
            ) from e

        calc = nml.UnivariateDriftCalculator(
            column_names=column_names,
            chunk_size=self.chunk_size,
        ).fit(reference_df[column_names])

        custom_objects: dict[str, Any] = {
            "drift_calculator": calc,
            "column_names": column_names,
            "chunk_size": self.chunk_size,
            "metric": self.metric,
            "forecaster_task_name": self.forecaster_task_name,
        }
        if self.input_spec is not None:
            custom_objects["input_spec"] = self.input_spec

        import bentoml
        bento = bentoml.picklable_model.save_model(
            self.bento_name,
            calc,
            custom_objects=custom_objects,
            signatures={"calculate": {"batchable": False}},
        )
        self._invalidate()
        return TrainResponse(
            model_tag=str(bento.tag),
            metrics={"reference_size": int(len(reference_df))},
        )

    def predict(self, req: PredictRequest) -> PredictResponse:
        if self.input_spec is not None:
            self.input_spec.validate(req.input_series)

        bento = self._load_bento(version=req.model_version)
        if bento is None:
            resolved = self._resolve_version(req.model_version)
            raise FileNotFoundError(
                f"no Bento {self.bento_name}:{resolved} in the local store; "
                f"POST /tasks/{self.name}/train first, or pin to an existing version"
            )
        self._verify_bento(bento)

        calc = bento.custom_objects["drift_calculator"]
        column_names: list[str] = bento.custom_objects["column_names"]
        metric: str = bento.custom_objects.get("metric", "jensen_shannon")

        analysis_df = pd.DataFrame(
            {col: req.input_series[col] for col in column_names if col in req.input_series}
        )
        if analysis_df.empty:
            raise ValueError(
                f"input_series carries none of the expected drift columns: {column_names}"
            )

        results = calc.calculate(analysis_df)
        chunks = results.filter(period="analysis").to_df()

        alert_cols = [
            (col, metric, "alert")
            for col in column_names
            if (col, metric, "alert") in chunks.columns
        ]
        any_alert = bool(chunks[alert_cols].any().any()) if alert_cols else False

        served = str(getattr(bento, "tag", "")).split(":")[-1] or None
        return PredictResponse(
            prediction={
                "drift_detected": any_alert,
                "n_chunks": int(len(chunks)),
                "metric": metric,
                "forecaster": self.forecaster_task_name,
            },
            model_version=served,
        )


def make_drift_prepare(value_col: str | None = None) -> Callable[[pd.DataFrame], dict]:
    """Build a prepare callable that returns ``{reference_df, drift_columns}``.

    Skips timestamp-like columns; respects ``value_col`` when given.
    """

    def prepare(df: pd.DataFrame) -> dict:
        cols = [c for c in df.columns if c.lower() not in {"time", "timestamp", "date"}]
        if value_col is not None:
            if value_col not in cols:
                raise ValueError(f"value_col {value_col!r} not in dataset; available: {cols}")
            cols = [value_col]
        ref = df[cols].astype(float).reset_index(drop=True)
        return {"reference_df": ref, "drift_columns": cols}

    return prepare
