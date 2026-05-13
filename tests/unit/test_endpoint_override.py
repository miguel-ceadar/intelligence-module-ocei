"""Per-request Prometheus endpoint override.

Gated by ``intelligence.telemetry.allow_endpoint_override`` (default
``False``) — when off, a request that sets ``data_source.endpoint``
must be rejected (otherwise an authenticated /train POST is an SSRF
probe surface). When on, the request-supplied endpoint replaces the
configured one for that single call. Auth (token / TLS) stays
configured — request-level token overrides are not in scope.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest import mock

import pandas as pd
import pytest

from intelligence.api.schemas import PrometheusDataSource
from intelligence.config.settings import (
    IntelligenceConfig,
    PrometheusConfig,
    TelemetryConfig,
)


def _maybe_field(model, name: str):
    if name not in model.model_fields:
        pytest.skip(f"{model.__name__}.{name} not implemented yet")


def test_prometheus_data_source_accepts_optional_endpoint_override():
    _maybe_field(PrometheusDataSource, "endpoint")
    desc = PrometheusDataSource(
        kind="prometheus",
        window="1h",
        step="1m",
        endpoint="https://other-prom.example:9090",
    )
    assert desc.endpoint == "https://other-prom.example:9090"


def test_prometheus_data_source_endpoint_defaults_to_none():
    _maybe_field(PrometheusDataSource, "endpoint")
    desc = PrometheusDataSource(kind="prometheus", window="1h", step="1m")
    assert desc.endpoint is None


def test_telemetry_config_allow_endpoint_override_defaults_off():
    """Off by default — SSRF defense. Pilots flip it on per deployment."""
    _maybe_field(TelemetryConfig, "allow_endpoint_override")
    cfg = TelemetryConfig(
        source="prometheus",
        prometheus=PrometheusConfig(endpoint="http://prom:9090"),
    )
    assert cfg.allow_endpoint_override is False


def test_loader_rejects_override_when_flag_off():
    """Sending ``data_source.endpoint`` without flipping the flag is a
    ``ValueError`` (which the API translates to 422 — the same shape as
    any other invalid request)."""
    _maybe_field(PrometheusDataSource, "endpoint")
    _maybe_field(TelemetryConfig, "allow_endpoint_override")
    from intelligence.tasks.loaders import build_loader_for_task

    cfg = IntelligenceConfig(
        telemetry=TelemetryConfig(
            source="prometheus",
            prometheus=PrometheusConfig(endpoint="http://configured:9090"),
            # allow_endpoint_override defaults to False
        ),
    )
    loader = build_loader_for_task(cfg, "cpu_forecast_arima", query="up")
    desc = PrometheusDataSource(
        kind="prometheus",
        window="1h",
        step="1m",
        endpoint="https://other:9090",
    )
    with pytest.raises(ValueError, match="allow_endpoint_override"):
        loader(desc)


def test_loader_uses_override_when_flag_on():
    """Flag on + request endpoint set: the loader builds a one-shot source
    against the override. The configured auth + tls settings carry over."""
    _maybe_field(PrometheusDataSource, "endpoint")
    _maybe_field(TelemetryConfig, "allow_endpoint_override")
    from intelligence.tasks.loaders import build_loader_for_task

    cfg = IntelligenceConfig(
        telemetry=TelemetryConfig(
            source="prometheus",
            prometheus=PrometheusConfig(
                endpoint="http://configured:9090",
                token_env="PROM_TOKEN",
                tls_skip_verify=True,
            ),
            allow_endpoint_override=True,
        ),
    )
    loader = build_loader_for_task(cfg, "cpu_forecast_arima", query="up")
    desc = PrometheusDataSource(
        kind="prometheus",
        window="1h",
        step="1m",
        endpoint="https://other-prom.internal:9090",
    )

    # Patch PrometheusSource.fetch_range so we don't actually call out;
    # capture the source that ends up serving the call.
    captured_endpoints: list[str] = []

    def fake_fetch_range(self, query, start, end, step):
        captured_endpoints.append(self.endpoint)
        return pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 50,
                "value": [0.5 + i * 0.001 for i in range(50)],
            }
        )

    with mock.patch(
        "intelligence.telemetry.PrometheusSource.fetch_range",
        new=fake_fetch_range,
    ):
        loader(desc)

    assert captured_endpoints == ["https://other-prom.internal:9090"]


def test_loader_falls_back_to_configured_endpoint_when_no_override():
    """No override in the request: the loader uses the configured endpoint
    even when the flag is on."""
    _maybe_field(PrometheusDataSource, "endpoint")
    _maybe_field(TelemetryConfig, "allow_endpoint_override")
    from intelligence.tasks.loaders import build_loader_for_task

    cfg = IntelligenceConfig(
        telemetry=TelemetryConfig(
            source="prometheus",
            prometheus=PrometheusConfig(endpoint="http://configured:9090"),
            allow_endpoint_override=True,
        ),
    )
    loader = build_loader_for_task(cfg, "cpu_forecast_arima", query="up")
    desc = PrometheusDataSource(kind="prometheus", window="1h", step="1m")

    captured: list[str] = []

    def fake_fetch_range(self, query, start, end, step):
        captured.append(self.endpoint)
        return pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 50,
                "value": [0.5 + i * 0.001 for i in range(50)],
            }
        )

    with mock.patch(
        "intelligence.telemetry.PrometheusSource.fetch_range",
        new=fake_fetch_range,
    ):
        loader(desc)

    assert captured == ["http://configured:9090"]
