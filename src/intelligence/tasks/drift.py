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
from pathlib import Path
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

    # ---- New manifest-based protocol --------------------------------------
    #
    # Drift sits at the task layer (no separate ``Model`` class) but
    # speaks the same ``fit / save_artifacts / load_artifacts`` shape
    # the per-algorithm models use. NannyML has no documented non-
    # pickle save for its calculator, so we persist the reference
    # window + the config and refit on load — one-time cost per
    # artefact load, then cached.

    def fit(self, components: dict) -> tuple[dict, dict]:
        """Build the runtime artefacts from a prepared reference frame.

        No actual sklearn-style fit happens here — fitting the NannyML
        calculator is deferred to :meth:`load_artifacts`, so the
        artefacts dict is just the reference data plus the parameters
        the calculator needs to refit later.
        """
        reference_df: pd.DataFrame = components["reference_df"]
        column_names: list[str] = list(
            components.get("drift_columns")
            or [c for c in reference_df.columns if c.lower() not in {"time", "timestamp", "date"}]
        )
        metrics = {"reference_size": len(reference_df)}
        artifacts = {
            "reference_df": reference_df,
            "column_names": column_names,
            "chunk_size": self.chunk_size,
            "metric": self.metric,
            "forecaster_task_name": self.forecaster_task_name,
            "model_metrics": metrics,
        }
        return artifacts, metrics

    def save_artifacts(self, artifacts: dict, dest: Path) -> dict[str, str]:
        """Persist the reference frame and config sidecar.

        Returns the ``role -> filename`` map for the manifest. The
        calculator itself isn't serialised — :meth:`load_artifacts`
        refits it from the persisted reference.
        """
        from intelligence.ml.artifact.sidecars import (
            save_input_spec,
            save_json,
        )

        ref: pd.DataFrame = artifacts["reference_df"]
        ref.to_parquet(dest / "reference.parquet")
        save_json(
            dest,
            "drift.json",
            {
                "column_names": list(artifacts["column_names"]),
                "chunk_size": int(artifacts["chunk_size"]),
                "metric": str(artifacts["metric"]),
                "forecaster_task_name": str(artifacts["forecaster_task_name"]),
            },
        )
        save_json(
            dest,
            "metrics.json",
            artifacts.get("model_metrics", {"reference_size": len(ref)}),
        )

        files: dict[str, str] = {
            "reference": "reference.parquet",
            "config": "drift.json",
            "metrics": "metrics.json",
        }
        spec = artifacts.get("input_spec")
        if spec is not None:
            save_input_spec(dest, spec)
            files["input_spec"] = "input_spec.json"
        return files

    def load_artifacts(self, src: Path) -> dict:
        """Inverse of :meth:`save_artifacts`. Refits the NannyML
        calculator from the persisted reference frame so predict has
        a ready-to-use ``drift_calculator`` in the returned dict.
        """
        try:
            import nannyml as nml
        except ImportError as e:  # pragma: no cover — import gate
            raise ImportError("drift task requires the `nannyml` package") from e

        from intelligence.ml.artifact.sidecars import (
            load_input_spec,
            load_json,
        )

        ref = pd.read_parquet(src / "reference.parquet")
        config = load_json(src, "drift.json")
        column_names: list[str] = list(config["column_names"])
        chunk_size = int(config["chunk_size"])

        calc = nml.UnivariateDriftCalculator(
            column_names=column_names,
            chunk_size=chunk_size,
        ).fit(ref[column_names])

        loaded: dict[str, Any] = {
            "drift_calculator": calc,
            "reference_df": ref,
            "column_names": column_names,
            "chunk_size": chunk_size,
            "metric": str(config["metric"]),
            "forecaster_task_name": str(config.get("forecaster_task_name", "")),
            "model_metrics": load_json(src, "metrics.json"),
        }
        if (src / "input_spec.json").exists():
            loaded["input_spec"] = load_input_spec(src)
        return loaded

    def train(self, req: TrainRequest) -> TrainResponse:
        from intelligence.ml.artifact import save_artifact

        components = self.data_loader(req.data_source)
        artifacts, metrics = self.fit(components)
        if self.input_spec is not None:
            artifacts["input_spec"] = self.input_spec

        saved = save_artifact(
            self.bento_name,
            "drift",
            lambda dest: self.save_artifacts(artifacts, dest),
        )
        self._invalidate()
        return TrainResponse(model_tag=saved.tag, metrics=metrics)

    def predict(self, req: PredictRequest) -> PredictResponse:
        if self.input_spec is not None:
            self.input_spec.validate(req.input_series)

        loaded, served_tag = self._load_drift_artifact(version=req.model_version)
        if loaded is None:
            resolved = self._resolve_version(req.model_version)
            raise FileNotFoundError(
                f"no Bento {self.bento_name}:{resolved} in the local store; "
                f"POST /tasks/{self.name}/train first, or pin to an existing version"
            )
        self._verify_artifact(loaded)

        calc = loaded["drift_calculator"]
        column_names: list[str] = loaded["column_names"]
        metric: str = loaded.get("metric", "jensen_shannon")

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

        served = str(served_tag).split(":")[-1] if served_tag else None
        return PredictResponse(
            prediction={
                "drift_detected": any_alert,
                "n_chunks": len(chunks),
                "metric": metric,
                "forecaster": self.forecaster_task_name,
            },
            model_version=served,
        )

    def _load_drift_artifact(self, version: str | None = None) -> tuple[dict | None, str | None]:
        """Drift override of :meth:`BaseTask._load_artifact` — calls
        ``self.load_artifacts`` (which is on this task, not on a Model)
        and caches the loaded dict under the resolved version."""
        from intelligence.ml.artifact import get_artifact_by_tag

        resolved = self._resolve_version(version)
        if resolved in self._cached_artifacts:
            return self._cached_artifacts[resolved]
        saved = get_artifact_by_tag(f"{self.bento_name}:{resolved}")
        if saved is None:
            return None, None
        loaded = self.load_artifacts(saved.path)
        self._cached_artifacts[resolved] = (loaded, saved.tag)
        return loaded, saved.tag


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
