"""Input-data drift detection via NannyML's ``UnivariateDriftCalculator``.

The calculator is fit on the reference distribution at train time
and applied to an input chunk at predict time, returning
``{"drift_detected": bool, ...}``.

``forecaster_task_name`` records which forecaster this drift task is
paired with; it's stored alongside the artifact for traceability.

The ``prepare`` callable must yield ``{"reference_df": pd.DataFrame,
"drift_columns": list[str]}``. ``make_drift_prepare`` produces the
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
    """NannyML-backed input-data drift detection.

    Inherits the lazy-load and readiness machinery from ``BaseTask``
    but overrides ``train`` and ``predict``: there is no per-algorithm
    ``Model`` here — the fitted calculator is the artifact.
    """

    forecaster_task_name: str = ""
    chunk_size: int = 6
    metric: str = "jensen_shannon"

    # NannyML's calculator has no pickle-free save, so we persist the
    # reference window plus the config and refit at load time (cached).

    def fit(self, components: dict) -> tuple[dict, dict]:
        """Bundle the reference frame and config into an artifacts dict.

        The NannyML calculator itself is fit at load time, so this
        function only captures what's needed to refit it later.
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
        """Persist the reference frame and config sidecar; return the
        ``role -> filename`` map for the manifest.
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
        """Read the reference frame and config, refit the NannyML
        calculator, and return a dict ready for ``predict``.
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
        """Variant of ``BaseTask._load_artifact`` that calls
        ``self.load_artifacts`` (drift owns its own load path)."""
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
