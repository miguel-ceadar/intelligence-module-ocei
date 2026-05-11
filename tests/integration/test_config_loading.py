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
          dataclay:
            enabled: false
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
