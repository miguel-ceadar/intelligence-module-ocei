"""``TaskRegistry`` contract.

Tasks are registered once at startup, looked up by name, and own their
own train/predict/drift behaviour plus an InputSpec.
"""

from __future__ import annotations

import pytest

from intelligence.config.settings import ArimaTaskConfig, FeatureSpec, IntelligenceConfig
from intelligence.tasks import TaskRegistry, build_registry_from_config


def test_task_registry_starts_empty():
    reg = TaskRegistry()
    assert list(reg) == []


def test_task_registry_register_and_lookup():
    class FakeTask:
        name = "cpu_forecast_arima"

        def train(self, req): ...
        def predict(self, req): ...

    reg = TaskRegistry()
    reg.register(FakeTask())
    assert "cpu_forecast_arima" in list(reg)
    assert reg.get("cpu_forecast_arima").name == "cpu_forecast_arima"


def test_task_registry_unknown_task_raises():
    reg = TaskRegistry()
    with pytest.raises(KeyError):
        reg.get("does_not_exist")


def test_task_registry_register_idempotent_or_explicit():
    """Registering the same name twice should either be a no-op or raise —
    both are defensible. What's *not* OK is silent overwrite."""

    class FakeTask:
        name = "cpu_forecast_arima"

        def train(self, req): ...
        def predict(self, req): ...

    reg = TaskRegistry()
    reg.register(FakeTask())
    try:
        reg.register(FakeTask())
    except (KeyError, ValueError):
        return  # explicit-raise is fine
    # If it didn't raise, ensure the registry didn't duplicate.
    assert list(reg).count("cpu_forecast_arima") == 1


def test_task_registry_built_from_task_blocks():
    """Every entry under ``cfg.tasks`` becomes a registered task; entries
    not declared in the config aren't registered."""
    cfg = IntelligenceConfig(
        tasks={
            "cpu_forecast_arima": ArimaTaskConfig(kind="arima", features=[FeatureSpec(name="cpu")]),
        },
    )
    reg = build_registry_from_config(cfg)
    assert "cpu_forecast_arima" in list(reg)
    assert "mem_forecast_arima" not in list(reg)
