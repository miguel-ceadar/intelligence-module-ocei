"""``TelemetrySource`` Protocol — the seam between data fetching and prep.

A telemetry source returns a ``pandas.DataFrame``. What goes in the
DataFrame is implementation-defined; downstream ``prepare`` callables
(see ``intelligence.tasks.loaders``) reshape it into model components.

The Protocol is structural — implementations don't need to inherit from
it. ``@runtime_checkable`` enables ``isinstance(src, TelemetrySource)``
for the readiness probe path. Method signatures match what callers use;
optional time-window args let static sources ignore them while PromQL
sources require them.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class TelemetrySource(Protocol):
    """Fetch data from somewhere. Return a DataFrame.

    ``query`` interpretation is source-specific:
      - ``StaticSource``: CSV filename, relative to the configured base dir.
      - ``PrometheusSource`` (phase-2 §3.2): PromQL expression.

    ``start`` / ``end`` / ``step`` are required for time-windowed sources
    (Prometheus) and ignored by sources that return a fixed dataset
    (StaticSource reads the whole CSV).
    """

    def fetch_range(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
        step: timedelta | None = None,
    ) -> pd.DataFrame: ...
