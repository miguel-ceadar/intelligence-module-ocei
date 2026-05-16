"""``StaticSource`` — CSV-backed ``TelemetrySource``.

Reads a CSV from a configured base directory and returns it as a
DataFrame. Time-window args are accepted but ignored — the file is what
it is. Used for demo/dev, tests, and any deployment without a Prometheus
to point at.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from intelligence.data import SAMPLES_DIR
from intelligence.utils.columns import TIMESTAMP_COLS


class StaticSource:
    """CSV-backed telemetry source.

    Attributes:
        base_dir: directory that ``query`` filenames are resolved against.
            Defaults to the package-bundled ``samples/`` directory.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else SAMPLES_DIR

    def fetch_range(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
        step: timedelta | None = None,
    ) -> pd.DataFrame:
        path = self.base_dir / query
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {query} (looked in {self.base_dir})")
        df = pd.read_csv(path).dropna()
        # Downstream prepares split train/test by row position and slide
        # windows along the row axis. Out-of-order CSVs silently produce
        # straddled splits; sort here so the row-position invariant
        # holds without each prepare having to re-sort.
        ts_col = next((c for c in df.columns if c.lower() in TIMESTAMP_COLS), None)
        if ts_col is not None:
            df = df.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
        return df

    def is_ready(self) -> tuple[bool, str]:
        if not self.base_dir.exists():
            return False, f"samples dir missing: {self.base_dir}"
        if not self.base_dir.is_dir():
            return False, f"samples path is not a directory: {self.base_dir}"
        return True, "ok"
