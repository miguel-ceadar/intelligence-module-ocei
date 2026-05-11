"""External-system adapters.

- ``dataclay_client`` — phase-1 centralisation of DataClay connection;
  phase-2 deletion target.
- (Phase 2 will add ``telemetry`` here for ``PrometheusSource`` /
  ``StaticSource``.)
"""

from intelligence.adapters.dataclay_client import (
    DataClayConfig,
    dataclay_client,
    from_legacy_args,
)

__all__ = ["DataClayConfig", "dataclay_client", "from_legacy_args"]
