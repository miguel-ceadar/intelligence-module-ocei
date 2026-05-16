"""Typed config layer — YAML schemas and env-var overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from intelligence import config


@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        """
        intelligence:
          model_repo:
            hf_enabled: false
          telemetry:
            source: static
          tasks:
            cpu_forecast_arima:
              kind: arima
              steps_back: 1
              features:
                - name: cpu
                  value_range: [0.0, 1.0]
        """.strip()
    )
    return p


def test_config_loads_from_yaml(config_yaml: Path):
    cfg = config.load_config(config_yaml)
    assert "cpu_forecast_arima" in cfg.intelligence.tasks
    task = cfg.intelligence.tasks["cpu_forecast_arima"]
    assert task.kind == "arima"
    assert [f.name for f in task.features] == ["cpu"]


def test_config_telemetry_default_is_static(config_yaml: Path):
    cfg = config.load_config(config_yaml)
    assert cfg.intelligence.telemetry.source == "static"


def test_config_unknown_kind_raises(tmp_path: Path):
    """Misconfigured ``kind`` should fail at startup, not at first request."""
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
        intelligence:
          tasks:
            mystery_task:
              kind: transformer
              features:
                - name: cpu
        """.strip()
    )
    with pytest.raises(ValidationError):
        config.load_config(p)


def test_config_drift_with_unknown_forecaster_raises(tmp_path: Path):
    """Cross-reference check: drift's `forecaster:` must name a defined task."""
    p = tmp_path / "bad-drift.yaml"
    p.write_text(
        """
        intelligence:
          tasks:
            cpu_drift:
              kind: drift
              forecaster: does_not_exist
              features:
                - name: cpu
        """.strip()
    )
    with pytest.raises(ValueError, match="forecaster"):
        config.load_config(p)


def test_config_drift_with_mismatched_features_raises(tmp_path: Path):
    """A drift detector that watches features different from its paired
    forecaster's produces meaningless dashboards — drift alarms on a
    signal the forecaster never sees. Force the feature names to match
    at load time."""
    p = tmp_path / "mismatched-drift.yaml"
    p.write_text(
        """
        intelligence:
          tasks:
            cpu_forecast:
              kind: arima
              features:
                - name: cpu
            cpu_drift:
              kind: drift
              forecaster: cpu_forecast
              features:
                - name: mem
        """.strip()
    )
    with pytest.raises(ValueError, match="features"):
        config.load_config(p)


def test_config_drift_with_matching_features_loads(tmp_path: Path):
    """Happy path for the cross-feature validator — drift and forecaster
    both list ``cpu`` and the config loads cleanly."""
    p = tmp_path / "matched-drift.yaml"
    p.write_text(
        """
        intelligence:
          tasks:
            cpu_forecast:
              kind: arima
              features:
                - name: cpu
            cpu_drift:
              kind: drift
              forecaster: cpu_forecast
              features:
                - name: cpu
        """.strip()
    )
    cfg = config.load_config(p)
    assert set(cfg.intelligence.tasks) == {"cpu_forecast", "cpu_drift"}


def test_appconfig_defaults_evaluate_env_at_call_not_import(monkeypatch):
    """``AppConfig()`` with no YAML should read env vars at *call*, not at
    module import. The old ``intelligence: IntelligenceConfig =
    IntelligenceConfig()`` default captured env at class-definition
    time, so an env var set later via monkeypatch was ignored. Use
    ``Field(default_factory=...)`` so each call re-evaluates."""
    monkeypatch.setenv("INTELLIGENCE_TELEMETRY__PROMETHEUS__ENDPOINT", "http://late-bound:9090")
    monkeypatch.setenv("INTELLIGENCE_TELEMETRY__SOURCE", "prometheus")

    cfg = config.load_config(None, validate=False)
    assert cfg.intelligence.telemetry.source == "prometheus"
    assert cfg.intelligence.telemetry.prometheus is not None
    assert cfg.intelligence.telemetry.prometheus.endpoint == "http://late-bound:9090"


def test_config_env_var_override(monkeypatch, tmp_path: Path):
    """Standard pydantic-settings: env vars override file values."""
    p = tmp_path / "prom.yaml"
    p.write_text(
        """
        intelligence:
          telemetry:
            source: prometheus
            prometheus:
              endpoint: http://from-file:9090
        """.strip()
    )
    monkeypatch.setenv("INTELLIGENCE_TELEMETRY__PROMETHEUS__ENDPOINT", "http://override:9090")
    cfg = config.load_config(p)
    assert cfg.intelligence.telemetry.prometheus.endpoint == "http://override:9090"


# ---- telemetry.prometheus -------------------------------------------------


def test_config_loads_prometheus_block(tmp_path: Path):
    p = tmp_path / "prom.yaml"
    p.write_text(
        """
        intelligence:
          telemetry:
            source: prometheus
            prometheus:
              endpoint: http://prom.monitoring.svc:9090
              token_env: PROM_TOKEN
              tls_skip_verify: false
          tasks:
            cpu_forecast_arima:
              kind: arima
              features:
                - name: cpu
                  query: 'avg(rate(node_cpu_seconds_total{mode!="idle"}[30s]))'
        """.strip()
    )
    cfg = config.load_config(p)
    telemetry = cfg.intelligence.telemetry
    assert telemetry.source == "prometheus"
    assert telemetry.prometheus is not None
    assert telemetry.prometheus.endpoint == "http://prom.monitoring.svc:9090"
    assert telemetry.prometheus.token_env == "PROM_TOKEN"
    # Per-feature query lives on the feature spec, not in a central queries dict.
    assert cfg.intelligence.tasks["cpu_forecast_arima"].features[0].query.startswith("avg(rate")


def test_config_rejects_prometheus_source_without_block(tmp_path: Path):
    """A config that says ``source: prometheus`` but omits the block must
    fail at load time — operators shouldn't discover the misconfig on the
    first prometheus request."""
    p = tmp_path / "bad-prom.yaml"
    p.write_text(
        """
        intelligence:
          telemetry:
            source: prometheus
        """.strip()
    )
    with pytest.raises(ValidationError):
        config.load_config(p)


def test_config_unknown_telemetry_source_rejected(tmp_path: Path):
    p = tmp_path / "bad-source.yaml"
    p.write_text(
        """
        intelligence:
          telemetry:
            source: otel
        """.strip()
    )
    with pytest.raises(ValidationError):
        config.load_config(p)
