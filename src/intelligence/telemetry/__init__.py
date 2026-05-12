"""Telemetry sources — fetch raw data from somewhere, return a DataFrame.

The ``TelemetrySource`` Protocol is the seam. Implementations:
  - ``StaticSource``: CSV-backed (demo / dev / tests).
  - ``PrometheusSource``: PromQL ``/api/v1/query_range`` (and Thanos).
"""

from intelligence.telemetry.base import TelemetrySource
from intelligence.telemetry.prometheus import PrometheusSource
from intelligence.telemetry.static import StaticSource

__all__ = ["PrometheusSource", "StaticSource", "TelemetrySource"]
