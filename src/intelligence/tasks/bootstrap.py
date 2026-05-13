"""Bootstrap auto-train on startup.

When ``cfg.tasks[name].bootstrap.auto_train_on_startup`` is true, the
service spawns a background coroutine on startup that calls
``task.train(...)`` against the configured data source. The task is
marked ``running`` while training, then ``complete`` (or ``failed``
with the error message). ``/readyz`` only returns 200 once every
bootstrap-required task is ``complete``.

This module is import-light: only schemas and stdlib. The actual
training runs on a worker thread via ``asyncio.to_thread`` so the
event loop stays responsive while the model fits.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from intelligence.api.schemas import (
    PrometheusDataSource,
    StaticDataSource,
    TrainRequest,
)

if TYPE_CHECKING:
    from intelligence.config.settings import BootstrapConfig, IntelligenceConfig
    from intelligence.tasks.base import BaseTask

logger = logging.getLogger(__name__)


def build_bootstrap_data_source(
    cfg: IntelligenceConfig,
    boot: BootstrapConfig,
) -> StaticDataSource | PrometheusDataSource:
    """Translate a ``BootstrapConfig`` into a concrete data source descriptor.

    Static mode requires ``dataset_name``; Prometheus mode requires
    ``window`` + ``step``. The chosen mode follows ``cfg.telemetry.source``.
    """
    if cfg.telemetry.source == "prometheus":
        if not boot.window or not boot.step:
            raise ValueError(
                "bootstrap requires telemetry.source=prometheus to also set "
                "window and step on the task's bootstrap block"
            )
        return PrometheusDataSource(kind="prometheus", window=boot.window, step=boot.step)

    if not boot.dataset_name:
        raise ValueError(
            "bootstrap requires telemetry.source=static to also set "
            "dataset_name on the task's bootstrap block"
        )
    return StaticDataSource(kind="static", name=boot.dataset_name)


async def bootstrap_task(task: BaseTask, cfg: IntelligenceConfig) -> None:
    """Run a single task's bootstrap, updating its state in place.

    No-op when the task has no ``tasks[name].bootstrap.auto_train_on_startup``
    entry or it's False. Exceptions during training are caught and
    surfaced via ``task.bootstrap_error``; they don't propagate out
    (this coroutine is fire-and-forget).
    """
    task_cfg = cfg.tasks.get(task.name)
    if task_cfg is None or not task_cfg.bootstrap.auto_train_on_startup:
        return

    try:
        ds = build_bootstrap_data_source(cfg, task_cfg.bootstrap)
    except ValueError as e:
        task.bootstrap_state = "failed"
        task.bootstrap_error = str(e)
        logger.error("task %s: bootstrap config invalid: %s", task.name, e)
        return

    req = TrainRequest(data_source=ds, model_parameters={})
    task.bootstrap_state = "running"
    logger.info("task %s: bootstrap starting", task.name)
    try:
        await asyncio.to_thread(task.train, req)
    except Exception as e:
        task.bootstrap_state = "failed"
        task.bootstrap_error = f"{type(e).__name__}: {e}"
        logger.exception("task %s: bootstrap failed", task.name)
        return

    task.bootstrap_state = "complete"
    logger.info("task %s: bootstrap complete", task.name)
