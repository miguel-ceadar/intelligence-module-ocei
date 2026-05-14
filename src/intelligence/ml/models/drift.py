"""NannyML-backed input-data drift detector implementing the ``Model``
protocol.

Drift's "artifact" is a fitted reference distribution rather than a
forecasting network — but the lifecycle (fit, save, load, predict) is
the same shape as ARIMA / XGB / LSTM, so the model slots into a
plain ``BaseTask`` the same way. ``BaseTask`` handles the
artifact-store plumbing; this module owns the drift-specific bits.

NannyML's ``UnivariateDriftCalculator`` has no documented non-pickle
save, so ``save_artifacts`` persists the reference DataFrame as parquet
plus a small JSON config, and ``load_artifacts`` refits the calculator
on load (cheap — one pass over the reference distribution).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_TIMESTAMP_COLS = {"time", "timestamp", "date"}


class DriftModel:
    """NannyML drift detector behind the ``Model`` protocol.

    ``has_drift = False`` — the flag on a model means "this forecaster
    carries a paired drift detector". This *is* the drift detector;
    it doesn't pair with another.
    """

    name = "drift"
    has_drift = False

    def __init__(
        self,
        chunk_size: int = 6,
        metric: str = "jensen_shannon",
        forecaster_task_name: str = "",
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.metric = metric
        self.forecaster_task_name = forecaster_task_name

    def fit(self, components: dict) -> tuple[dict, dict]:
        """Bundle the reference frame and config into an artifacts dict.

        The NannyML calculator itself is refit at load time so we don't
        rely on pickle. ``fit`` therefore only captures what's needed
        to reconstruct it.
        """
        reference_df: pd.DataFrame = components["reference_df"]
        column_names: list[str] = list(
            components.get("drift_columns")
            or [c for c in reference_df.columns if c.lower() not in _TIMESTAMP_COLS]
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
        from intelligence.ml.artifact.sidecars import save_input_spec, save_json

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
        """Read the reference + config and refit the NannyML calculator.

        Returns the same dict shape ``fit`` emits plus the live
        ``drift_calculator`` and (when persisted) the ``input_spec``.
        """
        # nannyml is the heaviest dep here — imported lazily so the
        # rest of the package doesn't pay its load cost.
        try:
            import nannyml as nml
        except ImportError as e:  # pragma: no cover — import gate
            raise ImportError("drift task requires the `nannyml` package") from e

        from intelligence.ml.artifact.sidecars import load_input_spec, load_json

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

    def predict(
        self,
        artifacts: dict,
        input_series: dict[str, list[float]],
        horizon: int = 1,
    ) -> dict[str, Any]:
        """Run the loaded calculator on the request chunk.

        ``horizon`` is accepted to fit the ``Model`` protocol signature
        and ignored — drift verdicts are about *now*, not the future.

        ``input_series`` must contain every column the artifact was
        fit on; missing columns raise rather than silently filter
        (the previous implementation intersected and returned a
        meaningless verdict over the surviving subset).
        """
        calc = artifacts["drift_calculator"]
        column_names: list[str] = list(artifacts["column_names"])
        metric: str = artifacts.get("metric", "jensen_shannon")
        forecaster: str = artifacts.get("forecaster_task_name", "")

        missing = [c for c in column_names if c not in input_series]
        if missing:
            raise ValueError(
                f"drift predict: input_series missing column(s) {missing!r}; "
                f"artifact expects {column_names!r}, got {list(input_series.keys())!r}"
            )

        analysis_df = pd.DataFrame({col: input_series[col] for col in column_names})
        results = calc.calculate(analysis_df)
        chunks = results.filter(period="analysis").to_df()

        alert_cols = [
            (col, metric, "alert")
            for col in column_names
            if (col, metric, "alert") in chunks.columns
        ]
        any_alert = bool(chunks[alert_cols].any().any()) if alert_cols else False

        return {
            "drift_detected": any_alert,
            "n_chunks": len(chunks),
            "metric": metric,
            "forecaster": forecaster,
        }


def make_drift_prepare(value_col: str | None = None) -> Callable[[pd.DataFrame], dict]:
    """Build a prepare callable that returns ``{reference_df, drift_columns}``.

    Drops timestamp-like columns; respects ``value_col`` when given (the
    single-column legacy path).
    """

    def prepare(df: pd.DataFrame) -> dict:
        cols = [c for c in df.columns if c.lower() not in _TIMESTAMP_COLS]
        if value_col is not None:
            if value_col not in cols:
                raise ValueError(f"value_col {value_col!r} not in dataset; available: {cols}")
            cols = [value_col]
        ref = df[cols].astype(float).reset_index(drop=True)
        return {"reference_df": ref, "drift_columns": cols}

    return prepare
