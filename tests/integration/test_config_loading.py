"""Phase-1 §2.5: typed config layer (pydantic-settings + config.yaml).

Replaces the conflated ``api_service_configs.json``. Static deployment
config is loaded once at startup; per-request overrides stay in JSON
request bodies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

config = pytest.importorskip("intelligence.config", reason="phase-1 §2.5 pending")


def _maybe(name: str):
    obj = getattr(config, name, None)
    if obj is None:
        pytest.skip(f"intelligence.config.{name} not implemented yet")
    return obj


@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        """
        intelligence:
          enabled_tasks:
            - cpu_forecast_arima
          mlflow:
            tracking_uri: http://localhost:5000
            auto_gc: false
          model_repo:
            hf_enabled: false
          telemetry:           # phase-2 placeholder; ignored in phase 1
            source: static
        """.strip()
    )
    return p


def test_config_loads_from_yaml(config_yaml: Path):
    load = _maybe("load_config")
    cfg = load(config_yaml)
    assert "cpu_forecast_arima" in cfg.intelligence.enabled_tasks


def test_config_telemetry_placeholder_present(config_yaml: Path):
    """Phase-1 ships the placeholder so the phase-2 add is purely additive."""
    load = _maybe("load_config")
    cfg = load(config_yaml)
    telemetry = getattr(cfg.intelligence, "telemetry", None)
    assert telemetry is not None
    assert getattr(telemetry, "source", None) == "static"


def test_config_unknown_task_in_enabled_raises(tmp_path: Path):
    """Misconfigured ``enabled_tasks`` should fail at startup, not at
    first request."""
    load = _maybe("load_config")
    p = tmp_path / "bad.yaml"
    p.write_text(
        "intelligence:\n  enabled_tasks: [does_not_exist]\n"
    )
    with pytest.raises(Exception):  # ValidationError or a project-specific subclass
        cfg = load(p)
        # Some implementations validate lazily; force the check.
        validate = getattr(cfg, "validate_against_registry", None)
        if validate is not None:
            validate()
        else:
            pytest.skip("validation hook not implemented yet")


def test_config_env_var_override(monkeypatch, config_yaml: Path):
    """Standard pydantic-settings: env vars override file values."""
    load = _maybe("load_config")
    monkeypatch.setenv("INTELLIGENCE_MLFLOW__TRACKING_URI", "http://override:5000")
    cfg = load(config_yaml)
    assert cfg.intelligence.mlflow.tracking_uri == "http://override:5000"


# ---- Phase 2: telemetry.prometheus -----------------------------------


def test_config_loads_prometheus_block(tmp_path: Path):
    load = _maybe("load_config")
    p = tmp_path / "prom.yaml"
    p.write_text(
        """
        intelligence:
          enabled_tasks: [cpu_forecast_arima]
          telemetry:
            source: prometheus
            prometheus:
              endpoint: http://prom.monitoring.svc:9090
              token_env: PROM_TOKEN
              tls_skip_verify: false
              queries:
                cpu_forecast_arima: 100 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100
        """.strip()
    )
    cfg = load(p)
    telemetry = cfg.intelligence.telemetry
    assert telemetry.source == "prometheus"
    assert telemetry.prometheus is not None
    assert telemetry.prometheus.endpoint == "http://prom.monitoring.svc:9090"
    assert telemetry.prometheus.token_env == "PROM_TOKEN"
    assert "cpu_forecast_arima" in telemetry.prometheus.queries


def test_config_rejects_prometheus_source_without_block(tmp_path: Path):
    """A config that says ``source: prometheus`` but omits the block must
    fail at load time — operators shouldn't discover the misconfig on the
    first prometheus request."""
    load = _maybe("load_config")
    p = tmp_path / "bad-prom.yaml"
    p.write_text(
        """
        intelligence:
          enabled_tasks: [cpu_forecast_arima]
          telemetry:
            source: prometheus
        """.strip()
    )
    with pytest.raises(Exception):  # ValidationError
        load(p)


def test_config_unknown_telemetry_source_rejected(tmp_path: Path):
    load = _maybe("load_config")
    p = tmp_path / "bad-source.yaml"
    p.write_text(
        """
        intelligence:
          telemetry:
            source: otel  # not yet supported
        """.strip()
    )
    with pytest.raises(Exception):
        load(p)
