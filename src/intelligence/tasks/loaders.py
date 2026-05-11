"""Static data loaders — phase 1 only.

Phase 2 will introduce ``intelligence.adapters.telemetry.PrometheusSource``
alongside these. Both will produce the same ``data_components`` shape so
``BaseTask`` doesn't care which side it came from.

Loaders are classes (not closures) so they can carry an ``is_ready()``
probe — the readiness endpoint asks each task's loader whether its
external dependencies are reachable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from intelligence.api.schemas import StaticDataSource

# Phase-1 dataset directory. Phase 2 makes this configurable + adds the
# Prometheus path; this constant goes away with the legacy oasis/ tree.
_LEGACY_DATASET_DIR = Path(__file__).resolve().parents[3] / "oasis" / "dataset"


class StaticCsvLoader:
    """Read a CSV from a configured base directory, prepare univariate
    train/test components.
    """

    def __init__(
        self,
        value_col: str | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.value_col = value_col
        self.base_dir = Path(base_dir) if base_dir is not None else _LEGACY_DATASET_DIR

    def __call__(self, source: StaticDataSource) -> dict:
        if not isinstance(source, StaticDataSource):
            raise ValueError(
                f"StaticCsvLoader expects StaticDataSource, got {type(source).__name__}"
            )
        path = self.base_dir / source.name
        if not path.exists():
            raise FileNotFoundError(
                f"dataset not found: {source.name} (looked in {self.base_dir})"
            )
        df = pd.read_csv(path).dropna()
        col = self.value_col or _autodetect_value_column(df)
        series = df[col].astype(float).values.reshape(-1, 1)
        split = int(len(series) * 0.8)
        scaler = MinMaxScaler().fit(series[:split])
        return {
            "X_train": scaler.transform(series[:split]),
            "X_test": scaler.transform(series[split:]),
            "y_train": series[:split].ravel(),
            "y_test": series[split:].ravel(),
            "scaler_obj": scaler,
        }

    def is_ready(self) -> tuple[bool, str]:
        if not self.base_dir.exists():
            return False, f"dataset dir missing: {self.base_dir}"
        if not self.base_dir.is_dir():
            return False, f"dataset path is not a directory: {self.base_dir}"
        return True, "ok"


def static_csv_loader(value_col: str | None = None, base_dir: Path | None = None) -> StaticCsvLoader:
    """Build a StaticCsvLoader. Thin wrapper kept so factory call sites read naturally."""
    return StaticCsvLoader(value_col=value_col, base_dir=base_dir)


def _autodetect_value_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() in {"time", "timestamp", "date"}:
            continue
        try:
            pd.to_numeric(df[col])
            return col
        except (ValueError, TypeError):
            continue
    raise ValueError(f"no numeric column found in dataset; columns={list(df.columns)}")
