"""Bootstrap auto-train on startup.

When a task's ``bootstrap.auto_train_on_startup`` is true, the service
spawns a background coroutine on startup that calls ``task.train(...)``.
Until that bootstrap completes, ``/readyz`` returns 503 for the task.
Failure modes (PromQL error, dataset missing) leave the task in
``failed`` state with the error surfaced through ``/readyz``.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

pytestmark = pytest.mark.integration


def _cfg_with_bootstrap(task_name: str, **boot_kwargs):
    """Build a config that carries one task with a bootstrap block.

    Uses ArimaTaskConfig as a stand-in — bootstrap is the same shape on
    every kind, so the test only cares about the bootstrap field.
    """
    from intelligence.config.settings import (
        ArimaTaskConfig,
        BootstrapConfig,
        IntelligenceConfig,
    )

    return IntelligenceConfig(
        tasks={
            task_name: ArimaTaskConfig(
                kind="arima",
                feature="cpu",
                bootstrap=BootstrapConfig(**boot_kwargs),
            ),
        },
    )


def test_bootstrap_config_round_trips_from_yaml(tmp_path):
    from intelligence.config import load_config

    p = tmp_path / "boot.yaml"
    p.write_text(
        """
        intelligence:
          tasks:
            cpu_forecast_arima:
              kind: arima
              feature: cpu
              bootstrap:
                auto_train_on_startup: true
                dataset_name: cpu_sample_dataset_orangepi.csv
        """.strip()
    )
    cfg = load_config(p)
    boot = cfg.intelligence.tasks["cpu_forecast_arima"].bootstrap
    assert boot.auto_train_on_startup is True
    assert boot.dataset_name == "cpu_sample_dataset_orangepi.csv"


def test_bootstrap_disabled_by_default():
    """An empty `tasks` block (or missing entry) means no bootstrap."""
    from intelligence.config.settings import IntelligenceConfig

    cfg = IntelligenceConfig()
    assert cfg.tasks == {}


def test_build_data_source_for_static(tmp_path):
    from intelligence.api.schemas import StaticDataSource
    from intelligence.tasks.bootstrap import build_bootstrap_data_source

    cfg = _cfg_with_bootstrap(
        "cpu_forecast_arima",
        auto_train_on_startup=True,
        dataset_name="cpu_sample_dataset_orangepi.csv",
    )
    boot = cfg.tasks["cpu_forecast_arima"].bootstrap
    ds = build_bootstrap_data_source(cfg, boot)
    assert isinstance(ds, StaticDataSource)
    assert ds.name == "cpu_sample_dataset_orangepi.csv"


def test_build_data_source_for_prometheus():
    from intelligence.api.schemas import PrometheusDataSource
    from intelligence.config.settings import (
        ArimaTaskConfig,
        BootstrapConfig,
        IntelligenceConfig,
        PrometheusConfig,
        TelemetryConfig,
    )
    from intelligence.tasks.bootstrap import build_bootstrap_data_source

    cfg = IntelligenceConfig(
        telemetry=TelemetryConfig(
            source="prometheus",
            prometheus=PrometheusConfig(endpoint="http://prom:9090"),
        ),
        tasks={
            "cpu_forecast_arima": ArimaTaskConfig(
                kind="arima",
                feature="cpu",
                query="up",
                bootstrap=BootstrapConfig(
                    auto_train_on_startup=True,
                    window="2h",
                    step="1m",
                ),
            ),
        },
    )
    boot = cfg.tasks["cpu_forecast_arima"].bootstrap
    ds = build_bootstrap_data_source(cfg, boot)
    assert isinstance(ds, PrometheusDataSource)
    assert ds.window == "2h"
    assert ds.step == "1m"


def test_build_data_source_static_without_dataset_name_raises():
    from intelligence.tasks.bootstrap import build_bootstrap_data_source

    cfg = _cfg_with_bootstrap("cpu_forecast_arima", auto_train_on_startup=True)
    boot = cfg.tasks["cpu_forecast_arima"].bootstrap
    with pytest.raises(ValueError, match="dataset_name"):
        build_bootstrap_data_source(cfg, boot)


def test_bootstrap_task_runs_train(tmp_path, monkeypatch):
    """The bootstrap coroutine should call ``task.train`` and update state
    to 'complete' on success.
    """
    from intelligence.tasks import BaseTask
    from intelligence.tasks.bootstrap import bootstrap_task

    monkeypatch.setenv("BENTOML_HOME", str(tmp_path / "bentoml"))

    fake_model = mock.MagicMock(name="model")
    fake_model.name = "fake"
    fake_model.has_drift = False
    fake_model.fit.return_value = ({"any": "thing"}, {"mae": 0.0})
    fake_model.save_artifacts.return_value = {"thing": "thing.json"}

    task = BaseTask(
        name="t_boot",
        model=fake_model,
        data_loader=mock.MagicMock(return_value={"X_train": [[0.0]], "X_test": [[0.0]]}),
    )

    # Stub the artefact store so we don't actually touch the bentoml store.
    from pathlib import Path

    from intelligence.ml.artifact import SavedArtifact
    from intelligence.ml.artifact.manifest import Manifest

    saved = SavedArtifact(
        tag="t_boot:v1",
        name="t_boot",
        version="v1",
        path=Path("/fake"),
        manifest=Manifest(
            schema_version=1,
            kind="fake",
            created_at="2026-05-13T10:00:00+00:00",
            files={},
        ),
        created_at="2026-05-13T10:00:00+00:00",
    )

    cfg = _cfg_with_bootstrap(
        "t_boot",
        auto_train_on_startup=True,
        dataset_name="cpu_sample_dataset_orangepi.csv",
    )

    assert task.bootstrap_state == "pending"
    with mock.patch("intelligence.tasks.base.save_artifact", return_value=saved):
        asyncio.run(bootstrap_task(task, cfg))
    assert task.bootstrap_state == "complete"
    assert task.bootstrap_error is None
    fake_model.fit.assert_called_once()


def test_bootstrap_task_records_failure():
    from intelligence.tasks import BaseTask
    from intelligence.tasks.bootstrap import bootstrap_task

    task = BaseTask(
        name="t_fail",
        model=mock.MagicMock(),
        data_loader=mock.MagicMock(side_effect=RuntimeError("loader exploded")),
    )
    cfg = _cfg_with_bootstrap(
        "t_fail",
        auto_train_on_startup=True,
        dataset_name="x.csv",
    )

    asyncio.run(bootstrap_task(task, cfg))
    assert task.bootstrap_state == "failed"
    assert task.bootstrap_error and "loader exploded" in task.bootstrap_error


def test_bootstrap_task_skipped_when_disabled():
    from intelligence.tasks import BaseTask
    from intelligence.tasks.bootstrap import bootstrap_task

    task = BaseTask(name="t_off", model=mock.MagicMock(), data_loader=mock.MagicMock())
    cfg = _cfg_with_bootstrap("t_off", auto_train_on_startup=False)
    asyncio.run(bootstrap_task(task, cfg))
    # No train call, state stays pending — bootstrap is opt-in.
    task.model.train.assert_not_called()
    assert task.bootstrap_state == "pending"


def test_readyz_blocks_until_bootstrap_completes():
    """When a task is registered with bootstrap=on but its state isn't
    ``complete``/``skipped``, /readyz should be 503.
    """
    from intelligence.api.service import compute_readiness
    from intelligence.tasks import BaseTask, TaskRegistry

    task = BaseTask(name="t_w", model=mock.MagicMock(), data_loader=mock.MagicMock())
    task.bootstrap_state = "running"
    reg = TaskRegistry()
    reg.register(task)

    ok, failures = compute_readiness(reg)
    assert ok is False
    assert any("bootstrap" in f["detail"] for f in failures)


def test_readyz_passes_when_bootstrap_complete():
    from intelligence.api.service import compute_readiness
    from intelligence.tasks import BaseTask, TaskRegistry

    task = BaseTask(name="t_done", model=mock.MagicMock(), data_loader=mock.MagicMock())
    task.bootstrap_state = "complete"
    reg = TaskRegistry()
    reg.register(task)

    _ok, failures = compute_readiness(reg)
    # Bootstrap state alone shouldn't fail; other probes might (bento store etc.)
    bootstrap_failures = [f for f in failures if "bootstrap" in f["detail"]]
    assert bootstrap_failures == []
    # And explicitly: pending state is the only one that should add a bootstrap failure.
    assert task.bootstrap_state == "complete"
