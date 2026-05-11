"""Task abstraction — see ``base.Task`` and ``base.TaskRegistry``.

Concrete task types live alongside (``forecast.ForecastTask``); the
catalog of named factories is in ``catalog``. Importing this package
does NOT trigger the catalog; that happens lazily when
``build_registry_from_config`` is called.
"""

from intelligence.tasks.base import (
    Task,
    TaskRegistry,
    build_registry_from_config,
    builtin_task_factory,
    register_builtin,
)

__all__ = [
    "Task",
    "TaskRegistry",
    "build_registry_from_config",
    "builtin_task_factory",
    "register_builtin",
]
