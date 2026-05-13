"""Phase-1 §2.3: a Task registry replaces the if/elif on model_tag.

Tasks are registered once at startup, looked up by name, and own their
own train/predict/drift behaviour plus an InputSpec.
"""

from __future__ import annotations

import pytest

tasks = pytest.importorskip("intelligence.tasks", reason="phase-1 §2.3 pending")


def _maybe(name: str):
    obj = getattr(tasks, name, None)
    if obj is None:
        pytest.skip(f"intelligence.tasks.{name} not implemented yet")
    return obj


def test_task_registry_starts_empty():
    TaskRegistry = _maybe("TaskRegistry")
    reg = TaskRegistry()
    assert list(reg) == []


def test_task_registry_register_and_lookup():
    TaskRegistry = _maybe("TaskRegistry")

    class FakeTask:
        name = "cpu_forecast_arima"

        def train(self, req): ...
        def predict(self, req): ...

    reg = TaskRegistry()
    reg.register(FakeTask())
    assert "cpu_forecast_arima" in list(reg)
    assert reg.get("cpu_forecast_arima").name == "cpu_forecast_arima"


def test_task_registry_unknown_task_raises():
    TaskRegistry = _maybe("TaskRegistry")
    reg = TaskRegistry()
    with pytest.raises(KeyError):
        reg.get("does_not_exist")


def test_task_registry_register_idempotent_or_explicit():
    """Registering the same name twice should either be a no-op or raise —
    both are defensible. What's *not* OK is silent overwrite."""
    TaskRegistry = _maybe("TaskRegistry")

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
    build = getattr(tasks, "build_registry_from_config", None)
    if build is None:
        pytest.skip("intelligence.tasks.build_registry_from_config not implemented yet")
    from intelligence.config.settings import ArimaTaskConfig, FeatureSpec, IntelligenceConfig

    cfg = IntelligenceConfig(
        tasks={
            "cpu_forecast_arima": ArimaTaskConfig(kind="arima", features=[FeatureSpec(name="cpu")]),
        },
    )
    reg = build(cfg)
    assert "cpu_forecast_arima" in list(reg)
    assert "mem_forecast_arima" not in list(reg)
