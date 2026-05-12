"""Per-kind task builders.

Each kind (``arima``, ``xgb``, ``lstm``, ``drift``) has a builder that
turns a typed config block into a concrete ``BaseTask`` (or subclass).
``build_registry_from_config`` dispatches over ``BUILDERS`` by the
config's ``kind`` field.

Adding a new kind is three local edits:

1. Add a new ``<Kind>TaskConfig`` to ``intelligence.config.settings``
   and include it in the ``TaskInstanceConfig`` union.
2. Add ``intelligence/tasks/builders/<kind>.py`` exporting
   ``build_<kind>_task(name, task_cfg, intelligence_cfg) -> BaseTask``.
3. Register it in the ``BUILDERS`` dict below.

Heavy imports stay inside each builder body so unconfigured kinds don't
pull dependencies they don't use.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from intelligence.tasks.builders.arima import build_arima_task
from intelligence.tasks.builders.drift import build_drift_task
from intelligence.tasks.builders.lstm import build_lstm_task
from intelligence.tasks.builders.xgb import build_xgb_task

if TYPE_CHECKING:
    from intelligence.config.settings import IntelligenceConfig, TaskInstanceConfig
    from intelligence.tasks.base import BaseTask

TaskBuilder = Callable[[str, "TaskInstanceConfig", "IntelligenceConfig"], "BaseTask"]

BUILDERS: dict[str, TaskBuilder] = {
    "arima": build_arima_task,
    "xgb": build_xgb_task,
    "lstm": build_lstm_task,
    "drift": build_drift_task,
}


def get_builder(kind: str) -> TaskBuilder:
    """Look up the builder for a config block's ``kind``. KeyError if
    the kind isn't registered — pydantic catches unknown kinds first at
    parse time, so reaching this branch means the BUILDERS dict and the
    TaskInstanceConfig union drifted apart."""
    try:
        return BUILDERS[kind]
    except KeyError as e:
        raise KeyError(
            f"no builder registered for kind {kind!r}; "
            f"known kinds: {sorted(BUILDERS)}"
        ) from e


__all__ = [
    "BUILDERS",
    "TaskBuilder",
    "build_arima_task",
    "build_drift_task",
    "build_lstm_task",
    "build_xgb_task",
    "get_builder",
]
