"""Unit tests for ``build_loader_for_task``.

The factory helper consults ``IntelligenceConfig.telemetry`` to decide
whether a task gets a CSV-backed loader or a PromQL-backed one. The
PromQL query for each task comes from the task's own config block —
callers (the per-kind builders) pass it through explicitly.
"""

from __future__ import annotations

import pytest

from intelligence.config.settings import IntelligenceConfig, PrometheusConfig, TelemetryConfig


def _cfg(**telemetry_kwargs) -> IntelligenceConfig:
    return IntelligenceConfig(telemetry=TelemetryConfig(**telemetry_kwargs))


def test_default_config_yields_static_loader():
    from intelligence.tasks.loaders import StaticCsvLoader, build_loader_for_task

    loader = build_loader_for_task(_cfg(), "cpu_forecast_arima")
    assert isinstance(loader, StaticCsvLoader)


def test_prometheus_config_yields_prometheus_loader():
    from intelligence.tasks.loaders import PrometheusLoader, build_loader_for_task

    cfg = _cfg(
        source="prometheus",
        prometheus=PrometheusConfig(endpoint="http://prom:9090"),
    )
    loader = build_loader_for_task(
        cfg,
        "cpu_forecast_arima",
        query="rate(node_cpu_seconds_total[5m])",
    )
    assert isinstance(loader, PrometheusLoader)
    assert loader.query == "rate(node_cpu_seconds_total[5m])"


def test_prometheus_missing_query_raises():
    from intelligence.tasks.loaders import build_loader_for_task

    cfg = _cfg(
        source="prometheus",
        prometheus=PrometheusConfig(endpoint="http://prom:9090"),
    )
    with pytest.raises(ValueError, match="no PromQL query"):
        build_loader_for_task(cfg, "cpu_forecast_arima")


def test_prometheus_loader_carries_endpoint_and_auth():
    from intelligence.tasks.loaders import build_loader_for_task
    from intelligence.telemetry import PrometheusSource

    cfg = _cfg(
        source="prometheus",
        prometheus=PrometheusConfig(
            endpoint="https://thanos.internal:10902",
            token_env="PROM_TOKEN",
            tls_skip_verify=True,
        ),
    )
    loader = build_loader_for_task(cfg, "cpu_forecast_arima", query="up")
    assert isinstance(loader.source, PrometheusSource)
    assert loader.source.endpoint == "https://thanos.internal:10902"
    assert loader.source.token_env == "PROM_TOKEN"
    assert loader.source.tls_skip_verify is True


def test_value_col_kwarg_is_accepted_for_both_loader_kinds():
    """The helper forwards ``value_col`` to whichever loader it builds so
    each task (e.g. ``mem_forecast_arima``) can normalise its column name.
    """
    from intelligence.tasks.loaders import build_loader_for_task

    static_loader = build_loader_for_task(_cfg(), "cpu_forecast_arima", value_col="CPU")
    assert static_loader is not None

    prom_cfg = _cfg(
        source="prometheus",
        prometheus=PrometheusConfig(endpoint="http://prom:9090"),
    )
    prom_loader = build_loader_for_task(
        prom_cfg,
        "cpu_forecast_arima",
        value_col="cpu",
        query="up",
    )
    assert prom_loader is not None
