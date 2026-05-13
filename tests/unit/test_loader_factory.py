"""Unit tests for ``build_loader_for_task``.

The factory helper consults ``IntelligenceConfig.telemetry`` to decide
whether a task gets a CSV-backed loader or a PromQL-backed one. The
PromQL queries for each task come from the task's own ``features:``
block — callers (the per-kind builders) pass them through explicitly
as parallel ``value_cols`` / ``queries`` lists.
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
        queries=["rate(node_cpu_seconds_total[5m])"],
    )
    assert isinstance(loader, PrometheusLoader)
    assert loader.queries == ["rate(node_cpu_seconds_total[5m])"]


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
    loader = build_loader_for_task(cfg, "cpu_forecast_arima", queries=["up"])
    assert isinstance(loader.source, PrometheusSource)
    assert loader.source.endpoint == "https://thanos.internal:10902"
    assert loader.source.token_env == "PROM_TOKEN"
    assert loader.source.tls_skip_verify is True


def test_value_cols_kwarg_is_accepted_for_both_loader_kinds():
    """The helper forwards ``value_cols`` to whichever loader it builds so
    each task can normalise its column names.
    """
    from intelligence.tasks.loaders import build_loader_for_task

    static_loader = build_loader_for_task(_cfg(), "cpu_forecast_arima", value_cols=["CPU"])
    assert static_loader is not None

    prom_cfg = _cfg(
        source="prometheus",
        prometheus=PrometheusConfig(endpoint="http://prom:9090"),
    )
    prom_loader = build_loader_for_task(
        prom_cfg,
        "cpu_forecast_arima",
        value_cols=["cpu"],
        queries=["up"],
    )
    assert prom_loader is not None


def test_prometheus_rejects_feature_with_missing_query():
    """If any feature's query is None in prom mode, fail at registry-build
    time so misconfigured multi-feature tasks don't blow up at /train."""
    from intelligence.tasks.loaders import build_loader_for_task

    cfg = _cfg(
        source="prometheus",
        prometheus=PrometheusConfig(endpoint="http://prom:9090"),
    )
    with pytest.raises(ValueError, match="no PromQL query"):
        build_loader_for_task(
            cfg,
            "mvar_task",
            value_cols=["cpu", "mem"],
            queries=["up", None],
        )
